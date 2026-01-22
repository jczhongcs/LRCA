
import os
import io
import lmdb
import pickle
import shutil
import logging
import numpy as np
import pandas as pd
import tmx_model_2
from tqdm import tqdm
import tmx_models3
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_score, recall_score,
    auc, precision_recall_curve
)

logger = logging.getLogger("Test")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


TEST_CSV = "Data/test_set_balanced.csv"
LMDB_PATH = r"Data/esm2_conv_lmdb_v3/features.lmdb"
MODEL_PATH_TMPL = "models/LRCA_fold_{fold}.pth"

BEST_OUT_PATH = "models/LRCA_best_by_AUPRC.pth"
OUT_METRICS_CSV = "Data/LRCA_eval_metrics_balanced.csv"

BATCH_SIZE = 128
NUM_WORKERS = 2
THRESHOLD = 0.5


class LMDBTestDataset(Dataset):
    def __init__(self, test_csv: str, lmdb_path: str):
        self.df = pd.read_csv(test_csv)
        self.headers = self.df["header"].astype(str).tolist()
        self.lmdb_path = lmdb_path
        self.env = None

    def _open(self):
        if self.env is not None:
            return
        self.env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )

    def __len__(self):
        return len(self.headers)

    def __getitem__(self, idx: int):
        self._open()
        h = self.headers[idx]
        key = h.encode("utf-8")
        with self.env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"Header not found in LMDB: {h}")

        item = pickle.loads(raw)
        vec = item["vec"]    # [D]
        tok = item["tok"]    # [K,D]
        mask = item["mask"]  # [K]
        return (vec, tok, mask), h

    def __getstate__(self):
        d = self.__dict__.copy()
        d["env"] = None
        return d


def _collate(batch):
    feats, headers = zip(*batch)
    vecs, toks, masks = zip(*feats)
    vec = torch.stack(vecs, 0)      # [B,D]
    tok = torch.stack(toks, 0)      # [B,K,D]
    mask = torch.stack(masks, 0)    # [B,K]
    return (vec, tok, mask), list(headers)


@torch.no_grad()
def predict(model, dataloader, threshold=0.5):
    model.eval()
    all_probs = []
    all_preds = []
    all_headers = []

    for (vec, tok, mask), headers in dataloader:
        vec = vec.to(DEVICE, non_blocking=True)
        tok = tok.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        logits = model(vec, tok, mask)  # [B,2]
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # [B,2]
        all_probs.extend(probs.tolist())
        all_headers.extend(headers)
        p1 = probs[:, 1]
        preds = (p1 > threshold).astype(np.int64)
        all_preds.extend(preds.tolist())

    return all_probs, all_preds, all_headers


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_CSV)
    header2label = dict(zip(test_df["header"].astype(str), test_df["label"].astype(int)))

    test_dataset = LMDBTestDataset(TEST_CSV, LMDB_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=_collate,
        drop_last=False,
    )

    (vec0, tok0, mask0), _ = test_dataset[0]
    vec_dim = int(vec0.shape[0])
    logger.info(f"Detected vec_dim from LMDB: {vec_dim}")

    auprcs, f1s, mccs, precs, recs, folds = [], [], [], [], [], []
    best_fold = None
    best_aupr = -1.0

    for fold in tqdm(range(1, 6), total=5):
        model_path = MODEL_PATH_TMPL.format(fold=fold)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = tmx_models3.DNNPredictor(
            vec_dim=1280,
            hidden_size=[512, 128],

            # token
            conv_channels=64,
            conv_kernel=7,
            local_num_layers=2,
            local_dilations=(1, 2),
            local_drop_path=0.15,
            token_drop=0.15,

            # pooling
            pool="mean",
            attn_pool_hidden=128,
            attn_pool_dropout=0.1,
            attn_softmax_fp32=True,
            tok_branch_drop=0.55,
            tok_pooled_dropout=0.55,
            fusion_dropout=0.30,
            feat_dim=32,
        ).to(DEVICE)

        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        probs, preds, headers = predict(model, test_loader, threshold=THRESHOLD)
        labels = np.array([int(header2label[h]) for h in headers], dtype=np.int64)
        preds = np.array(preds, dtype=np.int64)

        pos_probs = [p[1] for p in probs]
        pr, rc, _ = precision_recall_curve(labels, pos_probs)
        aupr = round(float(auc(rc, pr)), 4)

        rec = round(float(recall_score(labels, preds)), 4)
        prec = round(float(precision_score(labels, preds)), 4)
        f1v = round(float(f1_score(labels, preds)), 4)
        mcc = round(float(matthews_corrcoef(labels, preds)), 4)

        folds.append(fold)
        auprcs.append(aupr)
        recs.append(rec)
        precs.append(prec)
        f1s.append(f1v)
        mccs.append(mcc)

        logger.info(f"[Fold {fold}] AUPRC={aupr}  Recall={rec}  Precision={prec}  F1={f1v}  MCC={mcc}")

        if aupr > best_aupr:
            best_aupr = aupr
            best_fold = fold

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Average
    folds.append("Average")
    auprcs.append(round(float(np.mean(auprcs)), 4))
    recs.append(round(float(np.mean(recs)), 4))
    precs.append(round(float(np.mean(precs)), 4))
    f1s.append(round(float(np.mean(f1s)), 4))
    mccs.append(round(float(np.mean(mccs)), 4))

    logger.info(f"Best model fold (by AUPRC): {best_fold}  best_AUPRC={best_aupr:.4f}")

    # copy best
    shutil.copy(MODEL_PATH_TMPL.format(fold=best_fold), BEST_OUT_PATH)
    logger.info(f"Copied best model -> {BEST_OUT_PATH}")

    out = pd.DataFrame({
        "fold": folds,
        "AUPRC": auprcs,
        "Recall": recs,
        "Precision": precs,
        "F1_score": f1s,
        "MCC": mccs,
    })
    out.to_csv(OUT_METRICS_CSV, index=False)
    logger.info(f"Saved metrics -> {OUT_METRICS_CSV}")
