from __future__ import annotations

import torch
from wificsi_exp.models.network import CSIEncoder, CSIClassifier, FullModel


def main() -> None:
    x = torch.randn(4, 3, 114, 500)
    model = FullModel(num_classes=6, embedding_dim=128, hidden_dim=128, dropout=0.3)
    out = model(x)
    print(out["logits"].shape, out["embeddings"].shape)


if __name__ == "__main__":
    main()
