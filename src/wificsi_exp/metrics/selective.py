from __future__ import annotations

import numpy as np


def selective_metrics(probs: np.ndarray, y_true: np.ndarray, embeddings: np.ndarray, prototypes: dict[int, np.ndarray], tau_conf: float, tau_dist: float) -> dict[str, float]:
    conf = probs.max(axis=1)
    dists = []
    proto_keys = sorted(prototypes.keys())
    proto_arr = np.stack([prototypes[k] for k in proto_keys], axis=0)
    for emb in embeddings:
        d = np.linalg.norm(proto_arr - emb[None, :], axis=1).min()
        dists.append(d)
    dists = np.array(dists)
    keep = (conf >= tau_conf) & (dists <= tau_dist)
    coverage = float(keep.mean())
    if keep.sum() == 0:
        return {"coverage": coverage, "selective_accuracy": 0.0}
    preds = probs.argmax(axis=1)
    selective_acc = float((preds[keep] == y_true[keep]).mean())
    return {"coverage": coverage, "selective_accuracy": selective_acc}
