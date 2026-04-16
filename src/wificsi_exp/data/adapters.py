from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GenericMetadataDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, root: str | Path, file_format: str = "npy", signal_key: str = "csi", label_to_id: dict[str, int] | None = None) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.root = Path(root)
        self.file_format = file_format
        self.signal_key = signal_key
        labels = self.metadata["label"].astype(str).tolist()
        self.label_to_id = label_to_id or {lab: i for i, lab in enumerate(sorted(set(labels)))}

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_signal(self, path: Path) -> np.ndarray:
        if self.file_format == "npy":
            arr = np.load(path)
        elif self.file_format == "npz":
            arr = np.load(path)[self.signal_key]
        else:
            raise ValueError(f"Unsupported file_format: {self.file_format}")
        return arr.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.metadata.iloc[idx]
        sample_path = self.root / str(row["sample_path"])
        x = self._load_signal(sample_path)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D CSI tensor (C,S,T), got {x.shape}")
        y = self.label_to_id[str(row["label"])]
        domain = str(row.get("domain", "default"))
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
            "domain": domain,
            "index": idx,
        }


class NTUFiDataset(GenericMetadataDataset):
    pass


class Widar3Dataset(GenericMetadataDataset):
    pass
