# tmx_dataset.py
import io
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import lmdb
import torch
import pandas as pd
from torch.utils.data import Dataset

CSV_PATH = Path(r"Data\sequence_dataset_v3_substrate_pocket_aug.csv")
LMDB_PATH = Path(r"Data\esm2_conv_lmdb_v3\features.lmdb")

LMDB_READAHEAD = True
LMDB_LOCK = False
LMDB_MAX_READERS = 2048


class LMDBESM2FeatureStore:
    """
    LMDB feature store
      key: header (utf-8)
      val: bytes
          A) pickle.dumps(item)
          B) torch.save(item)
      item = {"vec": Tensor[D], "tok": Tensor[K,D], "mask": Tensor[K], "L": int}
    """

    def __init__(self):
        if not LMDB_PATH.exists():
            raise FileNotFoundError(f"LMDB not found: {LMDB_PATH}")

        self._env: Optional[lmdb.Environment] = None
        self._subdir = self._detect_subdir(LMDB_PATH)

    @staticmethod
    def _detect_subdir(path: Path) -> bool:
        if path.is_dir():
            return True
        if path.exists() and path.is_dir() and (path / "data.mdb").exists():
            return True
        return False

    def _open_env(self):
        if self._env is not None:
            return

        self._env = lmdb.open(
            str(LMDB_PATH),
            subdir=self._subdir,
            readonly=True,
            lock=LMDB_LOCK,
            readahead=LMDB_READAHEAD,
            meminit=False,
            max_readers=LMDB_MAX_READERS,
        )

    @staticmethod
    def _decode_item(raw: bytes) -> Dict[str, Any]:
        try:
            item = pickle.loads(raw)
            if isinstance(item, dict):
                return item
        except Exception:
            pass
        buf = io.BytesIO(raw)
        item = torch.load(buf, map_location="cpu")
        if not isinstance(item, dict):
            raise TypeError(f"LMDB item must be dict, got {type(item)}")
        return item

    def get_item(self, header: str) -> Dict[str, Any]:
        self._open_env()
        assert self._env is not None

        key = str(header).encode("utf-8")
        with self._env.begin(write=False) as txn:
            raw = txn.get(key)

        if raw is None:
            raise KeyError(f"Header not found in LMDB: {header}")

        item = self._decode_item(raw)
        if "vec" not in item or not torch.is_tensor(item["vec"]):
            raise KeyError(f"LMDB item missing Tensor 'vec' for header={header}")
        item["vec"] = item["vec"].contiguous()

        if "tok" in item:
            if not torch.is_tensor(item["tok"]):
                raise TypeError(f"'tok' must be Tensor for header={header}")
            item["tok"] = item["tok"].contiguous()

        if "mask" in item:
            if not torch.is_tensor(item["mask"]):
                raise TypeError(f"'mask' must be Tensor for header={header}")
            if item["mask"].dtype != torch.bool:
                item["mask"] = (item["mask"] > 0)
            item["mask"] = item["mask"].contiguous()

        return item
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_env"] = None
        return d


class SequenceDataset(Dataset):
    """
    Train Dataset：

    mode:
      - "vec":  Return (vec, y, header)
      - "conv": Return ((vec, tok, mask), y, header)
    """

    def __init__(self, fold: int, mode: str = "conv"):
        super().__init__()

        if not CSV_PATH.exists():
            raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

        self.dataset = pd.read_csv(CSV_PATH)
        if "header" not in self.dataset.columns or "label" not in self.dataset.columns:
            raise ValueError(f"CSV must contain header,label. got: {list(self.dataset.columns)}")

        self.headers = self.dataset["header"].astype(str).tolist()
        self.labels = self.dataset["label"].astype(int).tolist()

        self.mode = mode.lower().strip()
        if self.mode not in ("vec", "conv"):
            raise ValueError(f"mode must be vec or conv, got {mode}")

        self.store = LMDBESM2FeatureStore()
        self.set_fold(fold)

    def set_fold(self, fold: int):
        key = f"dataset_fold_{fold}"
        if key not in self.dataset.columns:
            raise ValueError(f"CSV missing fold column: {key}")

        self.train_ids = self.dataset[self.dataset[key] == "train"].index.values.tolist()
        self.val_ids = self.dataset[self.dataset[key] == "val"].index.values.tolist()
        self.test_ids = self.dataset[self.dataset[key] == "test"].index.values.tolist()

    def __getitem__(self, idx: int):
        header = self.headers[idx]
        y = torch.tensor([self.labels[idx]], dtype=torch.long)

        feat = self.store.get_item(header)
        vec = feat["vec"]

        if self.mode == "vec":
            return vec, y, header

        # conv mode
        if "tok" not in feat or "mask" not in feat:
            raise KeyError(f"conv mode requires tok/mask, but missing for header={header}")

        tok = feat["tok"]
        mask = feat["mask"]
        return (vec, tok, mask), y, header

    def __len__(self):
        return len(self.dataset)

