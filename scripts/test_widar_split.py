from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class WidarSplitDataset(Dataset):
    def __init__(self, split_csv: str | Path):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)

        labels = sorted(self.df["label_id"].dropna().astype(int).unique().tolist())
        self.label_to_index = {lab: i for i, lab in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.loadtxt(row["path"], delimiter=",", dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(x).float().unsqueeze(0)   # [1, H, W]
        y = self.label_to_index[int(row["label_id"])]

        meta = {
            "user": int(row["user"]),
            "label_name": row["label_name"],
            "filename": row["filename"],
        }
        return x, y, meta


def main():
    train_csv = Path("datasets/Widardata/splits_loso_user/train_user_1.csv")
    test_csv = Path("datasets/Widardata/splits_loso_user/test_user_1.csv")

    train_ds = WidarSplitDataset(train_csv)
    test_ds = WidarSplitDataset(test_csv)

    print(f"train len = {len(train_ds)}")
    print(f"test len  = {len(test_ds)}")

    x, y, meta = train_ds[0]
    print("sample train shape =", tuple(x.shape))
    print("sample train y     =", y)
    print("sample train meta  =", meta)

    x2, y2, meta2 = test_ds[0]
    print("sample test shape  =", tuple(x2.shape))
    print("sample test y      =", y2)
    print("sample test meta   =", meta2)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    batch_x, batch_y, batch_meta = next(iter(train_loader))

    print("batch_x.shape =", tuple(batch_x.shape))
    print("batch_y.shape =", tuple(batch_y.shape))
    print("batch users example =", batch_meta["user"][:5])


if __name__ == "__main__":
    main()