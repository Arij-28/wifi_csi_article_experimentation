from __future__ import annotations

import torch
from torch import nn


class CSIEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        z = self.head(h)
        return z


class CSIClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class FullModel(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = CSIEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = CSIClassifier(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z)
        return {"embeddings": z, "logits": logits}
