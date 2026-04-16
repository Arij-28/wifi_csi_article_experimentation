from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from wificsi_exp.metrics.classification import accuracy_and_macro_f1
from wificsi_exp.metrics.calibration import expected_calibration_error, brier_score
from wificsi_exp.metrics.selective import selective_metrics


@torch.no_grad()
def evaluate_model(model, loader, selective_cfg: dict | None = None, prototype_bank: dict[int, torch.Tensor] | None = None):
    model.eval()
    all_probs = []
    all_preds = []
    all_y = []
    all_emb = []
    for batch in loader:
        out = model(batch["x"])
        probs = F.softmax(out["logits"], dim=1)
        preds = probs.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_y.append(batch["y"].cpu().numpy())
        all_emb.append(out["embeddings"].cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    emb = np.concatenate(all_emb, axis=0)

    metrics = accuracy_and_macro_f1(y_true, preds)
    metrics["ece"] = expected_calibration_error(probs, y_true)
    metrics["brier"] = brier_score(probs, y_true)

    if selective_cfg and selective_cfg.get("selective_prediction", False) and prototype_bank:
        proto_np = {k: v.detach().cpu().numpy() for k, v in prototype_bank.items()}
        metrics.update(selective_metrics(probs, y_true, emb, proto_np, selective_cfg["tau_conf"], selective_cfg["tau_dist"]))
    return metrics
