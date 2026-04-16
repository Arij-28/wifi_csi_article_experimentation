from __future__ import annotations

import torch
from torch import optim

from wificsi_exp.data.builders import build_dataloaders
from wificsi_exp.models.network import FullModel


def build_all(cfg: dict):
    train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg)
    model = FullModel(
        num_classes=cfg["model"]["num_classes"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    return {
        "model": model,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
    }
