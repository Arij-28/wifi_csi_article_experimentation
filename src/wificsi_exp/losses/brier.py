from __future__ import annotations

import torch
import torch.nn.functional as F


def brier_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))
