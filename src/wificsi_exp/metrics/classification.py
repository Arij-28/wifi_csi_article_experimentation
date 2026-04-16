from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy_and_macro_f1(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
