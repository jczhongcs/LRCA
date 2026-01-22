import os
import sys
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

import torch
import lmdb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from esm import FastaBatchedDataset, pretrained



ALIGN_TO_CSV = True
CSV_PATH = Path(r"Data\sequence_dataset_v3_substrate_pocket_aug.csv")

TRAIN_NEG_FASTA = Path(r"Data\non_test_set_neg_all.fasta")
TEST_NEG_FASTA  = Path(r"Data\test_set_neg_all.fasta")
POS_FASTA       = Path(r"Data\positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta")

OUT_DIR = Path(r"Data\esm2_conv_lmdb_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LMDB_PATH = OUT_DIR / "features.lmdb"
INDEX_CSV = OUT_DIR / "index.csv"

MODEL_LOCATION = "esm2_t33_650M_UR50D"
DEVICE = "cuda:0"  # "cpu"

TRUNC_LEN = 1075
TOKS_PER_BATCH = 1400

# Conv-ready tok
K_FIXED = 256

# dtype
VEC_DTYPE = torch.float32
TOK_DTYPE = torch.float16
LMDB_MAP_SIZE_GB = 110

# LMDB commit
COMMIT_EVERY = 1000

PRINT_EVERY_BATCH = 10
PRINT_EVERY_ADDED = 200



def load_csv_headers(csv_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "header" not in r.fieldnames:
            raise ValueError(f"CSV loss header ,the truefiedl: {r.fieldnames}")
        for row in r:
            h = (row.get("header") or "").strip()
            if h:
                ids.add(h)
    return ids

def fasta_iter(path: Path) -> Iterable[Tuple[str, str]]:
    header: Optional[str] = None
    seq_chunks: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)

def bin_pool_tokens(aa: torch.Tensor, k_fixed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    aa: [L_i, D] on device
    return:
      tok:  [K, D] on device
      mask: [K]    on device (bool)
    """
    Li, D = aa.shape
    tok = aa.new_zeros((k_fixed, D))
    mask = torch.zeros((k_fixed,), dtype=torch.bool, device=aa.device)
    for b in range(k_fixed):
        s = (b * Li) // k_fixed
        e = ((b + 1) * Li) // k_fixed
        if e > s:
            tok[b] = aa[s:e].mean(dim=0)
            mask[b] = True
    return tok, mask

def get_done_ids_from_index(index_csv: Path) -> Set[str]:

    done: Set[str] = set()
    if not index_csv.exists():
        return done

    with open(index_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if not row:
                continue
            if first:
                first = False
                if row[0].strip().lower() == "header":
                    continue
            done.add(row[0])
    return done

def append_index_rows(index_csv: Path, headers: List[str]):
    write_header = not index_csv.exists()
    with open(index_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["header", "done"])
        for h in headers:
            w.writerow([h, 1])

def open_lmdb(lmdb_path: Path) -> lmdb.Environment:
    """
     LMDB（subdir=False）
    """
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        map_size=LMDB_MAP_SIZE_GB * 1024**3,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    return env



@torch.no_grad()
def process_fasta(
    fasta_path: Path,
    want_ids: Set[str],
    model,
    alphabet,
    device: str,
    env: lmdb.Environment,
    state: dict,
):
    """
    state ：
      - total_added: int
      - total_target: int
    """
    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNC_LEN),
        batch_sampler=batches,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    repr_layer = model.num_layers
    repr_layers = [repr_layer]

    txn = env.begin(write=True)
    pending_index_headers: List[str] = []

    for batch_idx, (labels, strs, toks) in enumerate(loader):
        if not want_ids:
            break

        if (batch_idx == 0) or ((batch_idx + 1) % PRINT_EVERY_BATCH == 0):
            print(f"  [BATCH] {batch_idx+1}/{len(batches)} | remaining in this fasta: {len(want_ids):,}")

        if device != "cpu":
            toks = toks.to(device=device, non_blocking=True)

        out = model(toks, repr_layers=repr_layers, return_contacts=False)
        reps = out["representations"][repr_layer]  # GPU if cuda

        for i, lab in enumerate(labels):
            hid = str(lab)
            if hid not in want_ids:
                continue

            if txn.get(hid.encode("utf-8")) is not None:
                want_ids.remove(hid)
                continue

            seq_len = min(TRUNC_LEN, len(strs[i]))
            aa = reps[i, 1: seq_len + 1]  # [L, D] on device

            vec_t = aa.mean(dim=0)  # [D]
            tok_t, mask_t = bin_pool_tokens(aa, K_FIXED)

            item = {
                "vec": vec_t.detach().to("cpu", dtype=VEC_DTYPE).contiguous(),
                "tok": tok_t.detach().to("cpu", dtype=TOK_DTYPE).contiguous(),
                "mask": mask_t.detach().to("cpu").contiguous(),
                "L": int(seq_len),
            }

            key = hid.encode("utf-8")
            val = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(key, val)

            pending_index_headers.append(hid)

            want_ids.remove(hid)
            state["total_added"] += 1

            if (state["total_added"] % PRINT_EVERY_ADDED) == 0:
                done = state["total_added"]
                total = state["total_target"]
                print(f"    [PROGRESS] done={done:,}/{total:,} | pending_commit={len(pending_index_headers):,}")

            # LOG
            if len(pending_index_headers) >= COMMIT_EVERY:
                meta = {
                    "k_fixed": K_FIXED,
                    "vec_dtype": str(VEC_DTYPE),
                    "tok_dtype": str(TOK_DTYPE),
                    "trunc_len": TRUNC_LEN,
                    "model": MODEL_LOCATION,
                }
                txn.put(b"__meta__", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

                txn.commit()
                append_index_rows(INDEX_CSV, pending_index_headers)
                pending_index_headers.clear()

                txn = env.begin(write=True)

                done = state["total_added"]
                total = state["total_target"]
                print(f"[CHECKPOINT] committed | done={done:,}/{total:,} | lmdb={LMDB_PATH}")

        # out
        del out, reps
        if device != "cpu":
            torch.cuda.empty_cache()

    if pending_index_headers:
        meta = {
            "k_fixed": K_FIXED,
            "vec_dtype": str(VEC_DTYPE),
            "tok_dtype": str(TOK_DTYPE),
            "trunc_len": TRUNC_LEN,
            "model": MODEL_LOCATION,
        }
        txn.put(b"__meta__", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

        txn.commit()
        append_index_rows(INDEX_CSV, pending_index_headers)
        pending_index_headers.clear()
    else:
        txn.commit()


def main():
    if ALIGN_TO_CSV:
        assert CSV_PATH.exists(), f"CSV is None: {CSV_PATH}"
    assert TRAIN_NEG_FASTA.exists(), f"FASTA is None: {TRAIN_NEG_FASTA}"
    assert TEST_NEG_FASTA.exists(), f"FASTA is None: {TEST_NEG_FASTA}"
    assert POS_FASTA.exists(), f"FASTA is None: {POS_FASTA}"

    if ALIGN_TO_CSV:
        target_ids = load_csv_headers(CSV_PATH)
        print(f"[CSV] headers: {len(target_ids):,}")
    else:
        target_ids = set()
        for p in [TRAIN_NEG_FASTA, TEST_NEG_FASTA, POS_FASTA]:
            for hid, _ in fasta_iter(p):
                target_ids.add(hid)
        print(f"[FASTA] total headers: {len(target_ids):,}")

    done_ids = get_done_ids_from_index(INDEX_CSV)
    print(f"[DONE] already in index: {len(done_ids):,}")

    want_ids = target_ids - done_ids
    print(f"[TODO] remaining to extract: {len(want_ids):,}")
    if not want_ids:
        print("Nothing to do.")
        return

    device = DEVICE if ("cuda" in DEVICE and torch.cuda.is_available()) else "cpu"
    print(f"[MODEL] device: {device}")
    model, alphabet = pretrained.load_model_and_alphabet(MODEL_LOCATION)
    model.eval()
    if device != "cpu":
        model = model.to(device)
        print("[MODEL] transferred to GPU")

    env = open_lmdb(LMDB_PATH)
    print(f"[LMDB] path={LMDB_PATH} | map_size={LMDB_MAP_SIZE_GB} GB")

    state = {
        "total_added": 0,
        "total_target": len(want_ids),
    }

    fasta_sources = [
        ("train_neg", TRAIN_NEG_FASTA),
        ("test_neg",  TEST_NEG_FASTA),
        ("pos",       POS_FASTA),
    ]

    for name, fasta_path in fasta_sources:
        if not want_ids:
            break
        print(f"[SCAN] {name}: {fasta_path} | remaining overall: {len(want_ids):,}")
        process_fasta(
            fasta_path=fasta_path,
            want_ids=want_ids,
            model=model,
            alphabet=alphabet,
            device=device,
            env=env,
            state=state,
        )

    env.sync()
    env.close()

    print(f"[DONE] total_added_this_run: {state['total_added']:,}")
    print(f"[REMAIN] still not found in FASTA (if any): {len(want_ids):,}")
    if want_ids:
        print("sample remaining (up to 30):", list(sorted(want_ids))[:30])

    print(f"[OUT] lmdb: {LMDB_PATH}")
    print(f"[OUT] index: {INDEX_CSV}")
    print(f"[CFG] K_FIXED={K_FIXED} | COMMIT_EVERY={COMMIT_EVERY} | TOK_DTYPE={TOK_DTYPE} | VEC_DTYPE={VEC_DTYPE}")


if __name__ == "__main__":
    main()
