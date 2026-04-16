from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_consistency_loss(embeddings: torch.Tensor, labels: torch.Tensor, domains: list[str]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    device = embeddings.device
    unique_labels = labels.unique().tolist()
    proto_map: dict[int, torch.Tensor] = {}
    total = torch.tensor(0.0, device=device)
    count = 0
    for lbl in unique_labels:
        lbl_mask = labels == lbl
        lbl_emb = embeddings[lbl_mask]
        lbl_domains = [domains[i] for i in torch.where(lbl_mask)[0].tolist()]
        domain_to_vecs = {}
        for vec, dom in zip(lbl_emb, lbl_domains):
            domain_to_vecs.setdefault(dom, []).append(vec)
        if len(domain_to_vecs) == 0:
            continue
        local_protos = []
        for dom, vecs in domain_to_vecs.items():
            local = torch.stack(vecs, dim=0).mean(dim=0)
            local_protos.append(local)
        global_proto = torch.stack(local_protos, dim=0).mean(dim=0)
        proto_map[int(lbl)] = global_proto.detach()
        for lp in local_protos:
            total = total + F.mse_loss(lp, global_proto)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=device), proto_map
    return total / count, proto_map
