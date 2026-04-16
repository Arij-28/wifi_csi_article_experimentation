from __future__ import annotations

from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from wificsi_exp.data.adapters import GenericMetadataDataset, NTUFiDataset, Widar3Dataset
from wificsi_exp.data.splits import make_splits_dataframe


def _get_dataset_class(name: str):
    if name == "ntu_fi":
        return NTUFiDataset
    if name == "widar3":
        return Widar3Dataset
    return GenericMetadataDataset


def build_dataloaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    meta = pd.read_csv(ds_cfg["metadata_csv"])
    if ds_cfg.get("split_column", "split") not in meta.columns:
        meta = make_splits_dataframe(meta, cfg, cfg["protocol"]["mode"])
    split_col = ds_cfg.get("split_column", "split")
    train_df = meta[meta[split_col] == "train"].copy()
    val_df = meta[meta[split_col] == "val"].copy()
    test_df = meta[meta[split_col] == "test"].copy()

    labels = sorted(meta[ds_cfg["label_column"]].astype(str).unique().tolist())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    for frame in (train_df, val_df, test_df):
        frame.rename(columns={ds_cfg["label_column"]: "label", ds_cfg["domain_column"]: "domain"}, inplace=True)

    cls = _get_dataset_class(ds_cfg["name"])
    common = dict(root=ds_cfg["root"], file_format=ds_cfg.get("file_format", "npy"), signal_key=ds_cfg.get("signal_key", "csi"), label_to_id=label_to_id)
    train_ds = cls(train_df, **common)
    val_ds = cls(val_df, **common)
    test_ds = cls(test_df, **common)

    def collate(batch):
        import torch
        xs = torch.stack([b["x"] for b in batch], dim=0)
        ys = torch.stack([b["y"] for b in batch], dim=0)
        domains = [b["domain"] for b in batch]
        return {"x": xs, "y": ys, "domain": domains}

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0), collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0), collate_fn=collate)
    return train_loader, val_loader, test_loader, labels
