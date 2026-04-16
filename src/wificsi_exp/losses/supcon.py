from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(len(labels), device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives = labels_eq & (~mask)

    loss = torch.tensor(0.0, device=device)
    valid = 0
    for i in range(len(labels)):
        pos_idx = positives[i]
        if pos_idx.sum() == 0:
            continue
        numerator = torch.logsumexp(sim[i][pos_idx], dim=0)
        denominator = torch.logsumexp(sim[i], dim=0)
        loss = loss - (numerator - denominator)
        valid += 1
    if valid == 0:
        return torch.tensor(0.0, device=device)
    return loss / valid
