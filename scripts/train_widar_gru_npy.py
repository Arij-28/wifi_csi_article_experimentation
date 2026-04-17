from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3


class WidarNpyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_index=None, normalize=True):
        self.df = df.reset_index(drop=True)
        self.normalize = normalize

        if label_to_index is None:
            labels = sorted(self.df["label_id"].dropna().astype(int).unique().tolist())
            self.label_to_index = {lab: i for i, lab in enumerate(labels)}
        else:
            self.label_to_index = label_to_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["npy_path"]).astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)

        # ici on garde [22, 400]
        x = torch.from_numpy(x).float()
        y = self.label_to_index[int(row["label_id"])]
        return x, y


class Widar_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gru = nn.GRU(400, 64, num_layers=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x attendu: [B, 22, 400]
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)   # [22, B, 400]
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs


def run_epoch(model, loader, optimizer=None):
    criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())

    acc = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return total_loss / total_n, acc, macro_f1


def main():
    meta_npy = Path("datasets/Widardata_npy/metadata_npy.csv")
    df = pd.read_csv(meta_npy)

    if "duplicate" in df.columns:
        df = df[df["duplicate"] == False].copy()  # noqa: E712

    # LOSO user = 1
    train_df = df[df["user"] != 1].copy()
    test_df = df[df["user"] == 1].copy()

    # mode debug rapide
    train_df = train_df.sample(n=min(3000, len(train_df)), random_state=42).reset_index(drop=True)
    test_df = test_df.sample(n=min(1000, len(test_df)), random_state=42).reset_index(drop=True)

    labels = sorted(train_df["label_id"].dropna().astype(int).unique().tolist())
    label_to_index = {lab: i for i, lab in enumerate(labels)}

    train_ds = WidarNpyDataset(train_df, label_to_index=label_to_index, normalize=True)
    test_ds = WidarNpyDataset(test_df, label_to_index=label_to_index, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Widar_GRU(num_classes=len(label_to_index)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"DEVICE = {DEVICE}")
    print(f"train size = {len(train_ds)}")
    print(f"test size  = {len(test_ds)}")
    print(f"num classes = {len(label_to_index)}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer=optimizer)
        test_loss, test_acc, test_f1 = run_epoch(model, test_loader, optimizer=None)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )


if __name__ == "__main__":
    main()