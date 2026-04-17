from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3


class WidarSplitDataset(Dataset):
    def __init__(self, split_csv: str | Path, label_to_index=None):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)

        if label_to_index is None:
            labels = sorted(self.df["label_id"].dropna().astype(int).unique().tolist())
            self.label_to_index = {lab: i for i, lab in enumerate(labels)}
        else:
            self.label_to_index = label_to_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.loadtxt(row["path"], delimiter=",", dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(x).float().unsqueeze(0)  # [1, 22, 400]
        y = self.label_to_index[int(row["label_id"])]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),   # [16, 11, 200]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # [32, 5, 100]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, optimizer=None):
    criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None

    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


def main():
    train_csv = Path("datasets/Widardata/splits_loso_user/train_user_1.csv")
    test_csv = Path("datasets/Widardata/splits_loso_user/test_user_1.csv")

    train_df = pd.read_csv(train_csv)
    labels = sorted(train_df["label_id"].dropna().astype(int).unique().tolist())
    label_to_index = {lab: i for i, lab in enumerate(labels)}

    train_ds = WidarSplitDataset(train_csv, label_to_index=label_to_index)
    test_ds = WidarSplitDataset(test_csv, label_to_index=label_to_index)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=len(label_to_index)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"DEVICE = {DEVICE}")
    print(f"train size = {len(train_ds)}")
    print(f"test size  = {len(test_ds)}")
    print(f"num classes = {len(label_to_index)}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer=optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, optimizer=None)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()