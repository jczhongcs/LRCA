import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from early_stopping import EarlyStopping
import Loss
import LRCA_Dataset
import LRCA_models
logger = logging.getLogger("Train")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _move_batch_to_device(batch_data):
    """
    batch_data from DataLoader:
      ((vec, tok, mask), labels, headers)
    """
    feat_pack, labels, headers = batch_data
    vec, tok, mask = feat_pack
    vec = vec.to(DEVICE, non_blocking=True)
    tok = tok.to(DEVICE, non_blocking=True)
    mask = mask.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    labels = labels.squeeze(dim=1).long()
    return vec, tok, mask, labels, headers


def train(
    model,
    train_dataloader,
    optimizer_model,
    optimizer_center,
    loss_ce,
    loss_center,
    lambda_center,
    epoch,
    profile_every: int = 50,
):
    model.train()
    train_loss = 0.0
    logger.info(f"Epoch ::: {epoch}")

    for batch_idx, batch_data in enumerate(train_dataloader):
        t0 = time.time()
        vec, tok, mask, labels, _ = _move_batch_to_device(batch_data)
        t1 = time.time()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()

        logits, feats = model(vec, tok, mask, return_feat=True)

        loss1 = loss_ce(logits, labels)
        loss2 = loss_center(feats, labels)
        loss = loss1 + lambda_center * loss2

        optimizer_model.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_model.step()
        optimizer_center.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.time()

        train_loss += loss.item()

        if profile_every > 0 and (batch_idx % profile_every == 0):
            logger.info(
                f"[PROFILE] batch={batch_idx:05d} "
                f"load+H2D={t1-t0:.3f}s  sync={t2-t1:.3f}s  step={t3-t2:.3f}s"
            )

    logger.info(f"Training loss: {round(train_loss / len(train_dataloader), 4)}")


@torch.no_grad()
def eval_metric(model, val_dataloader):
    model.eval()
    predictions = []
    test_labels = []

    for batch_data in val_dataloader:
        vec, tok, mask, labels, _ = _move_batch_to_device(batch_data)

        test_labels.extend(labels.detach().cpu().numpy().tolist())

        logits = model(vec, tok, mask)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1).astype(np.int64)
        predictions.extend(preds.tolist())

    return predictions, test_labels


def _build_loader(
    ds_subset: Subset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
):
    use_cuda = (DEVICE.type == "cuda")
    return DataLoader(
        ds_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        drop_last=False,
    )


if __name__ == "__main__":
    LR = 2e-4
    BATCH_SIZE = 64
    NUM_FOLDS = 5
    NUM_EPOCHS = 120
    LAMBDA_CENTER = 0
    NUM_WORKERS_TRAIN = 2
    NUM_WORKERS_EVAL = 1
    PREFETCH_FACTOR = 2
    os.makedirs("models", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True


    dataset = dataset.SequenceDataset(fold=1, mode="conv")
    test_dataset = Subset(dataset, indices=dataset.test_ids)
    test_dataloader = _build_loader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        prefetch_factor=PREFETCH_FACTOR,
    )

    for i in range(NUM_FOLDS):
        logger.info(f"-------Fold {i + 1}-------")
        dataset.set_fold(i + 1)

        train_dataset = Subset(dataset, indices=dataset.train_ids)
        val_dataset = Subset(dataset, indices=dataset.val_ids)

        train_dataloader = _build_loader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS_TRAIN,
            prefetch_factor=PREFETCH_FACTOR,
        )
        val_dataloader = _build_loader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS_EVAL,
            prefetch_factor=PREFETCH_FACTOR,
        )
        pos = 0
        neg = 0
        for idx in dataset.train_ids:
            _, y, _ = dataset[idx]
            yy = int(y.item())
            if yy == 1:
                pos += 1
            else:
                neg += 1
        logger.info(f"Fold {i + 1} train pos={pos}, neg={neg}")


        model = LRCA_models.DNNPredictor(
            vec_dim=1280,
            hidden_size=[512, 128],
            conv_channels=64,
            conv_kernel=7,
            local_num_layers=2,
            local_dilations=(1, 2),
            local_drop_path=0.15,
            token_drop=0.25,
            use_layernorm=True,
            pool="mean",
            attn_pool_hidden=128,
            attn_pool_dropout=0.2,
            attn_softmax_fp32=True,

            tok_branch_drop=0.55,
            tok_pooled_dropout=0.55,
            fusion_dropout=0.30,
            feat_dim=32,
        ).to(DEVICE)

        loss_ce = nn.CrossEntropyLoss()
        loss_center = Loss.CenterLoss(num_classes=2, feat_dim=32, device=DEVICE)

        optimizer_model = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        optimizer_center = Adam(loss_center.parameters(), lr=LR * 0.5)

        stopper = EarlyStopping(
            mode="higher",
            patience=5,
            filename=f"../models/LRCA_fold_{i + 1}.pth",
        )

        scheduler = ReduceLROnPlateau(
            optimizer_model,
            factor=0.2,
            patience=3,
            min_lr=5e-6,
        )

        logger.info("-------Starting training-------")
        for ne in range(NUM_EPOCHS):
            train(
                model,
                train_dataloader,
                optimizer_model,
                optimizer_center,
                loss_ce,
                loss_center,
                LAMBDA_CENTER,
                ne + 1,
                profile_every=50,
            )

            val_preds, val_labels = eval_metric(model, val_dataloader)
            f1 = f1_score(val_labels, val_preds)

            early_stop = stopper.step(f1, model)
            logger.info(f"Validation F1-score: {round(f1, 4)}")

            if early_stop:
                logger.info("Early stopped!")
                break

            scheduler.step(f1)


