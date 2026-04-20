from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n < 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# =========================================================
# Dataset / model (same as training/eval scripts)
# =========================================================

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

        x = torch.from_numpy(x).float()
        y = self.label_to_index[int(row["label_id"])]
        domain = int(row["user"])
        return x, y, domain


class BiGRUProto(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=400,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)  # [22, B, 400]
        _, ht = self.gru(x)
        feat_fwd = ht[-2]
        feat_bwd = ht[-1]
        feat = torch.cat([feat_fwd, feat_bwd], dim=1)
        logits = self.fc(feat)
        return logits, feat


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for x, y, _ in loader:
        x = x.to(device)
        logits, _ = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return logits, targets


# =========================================================
# Temperature scaling
# =========================================================

class ModelWithTemperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # T=1 at init

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)


def fit_temperature(val_logits: torch.Tensor, val_targets: torch.Tensor, device: str):
    scaler = ModelWithTemperature().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    logits = val_logits.to(device)
    targets = val_targets.to(device)

    optimizer = optim.LBFGS([scaler.log_temperature], lr=0.01, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler.temperature_scale(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


# =========================================================
# Reliability diagram helpers
# =========================================================

def compute_reliability_bins(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_acc = []
    bin_conf = []
    bin_frac = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        center = 0.5 * (lo + hi)
        bin_centers.append(center)

        if mask.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_frac.append(0.0)
        else:
            bin_acc.append(float(acc[mask].mean()))
            bin_conf.append(float(conf[mask].mean()))
            bin_frac.append(float(mask.mean()))

    return {
        "bin_centers": np.array(bin_centers),
        "bin_acc": np.array(bin_acc),
        "bin_conf": np.array(bin_conf),
        "bin_frac": np.array(bin_frac),
    }


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


# =========================================================
# Load run and reconstruct validation / test
# =========================================================

def load_run_probs(summary_json: Path, metadata_path: str, batch_size: int, num_workers: int, device: str):
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    user_test = int(summary["user_test"])
    subset_train = int(summary["subset_train"])
    subset_test = int(summary["subset_test"])
    seed = int(summary["seed"])
    val_ratio = float(summary.get("val_ratio", 0.1))

    ckpt_path = summary_json.with_name(summary_json.stem.replace("_summary", "_best") + ".pt")

    df = pd.read_csv(metadata_path)
    if "duplicate" in df.columns:
        df = df[df["duplicate"] == False].copy()  # noqa: E712

    source_df = df[df["user"] != user_test].copy()
    test_df = df[df["user"] == user_test].copy()

    source_df = subset_df(source_df, subset_train, seed)
    test_df = subset_df(test_df, subset_test, seed)

    train_df, val_df = train_test_split(
        source_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=source_df["label_id"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    labels = sorted(train_df["label_id"].dropna().astype(int).unique().tolist())
    label_to_index = {lab: i for i, lab in enumerate(labels)}

    val_ds = WidarNpyDataset(val_df, label_to_index=label_to_index, normalize=True)
    test_ds = WidarNpyDataset(test_df, label_to_index=label_to_index, normalize=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BiGRUProto(
        num_classes=len(label_to_index),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    val_logits, val_targets = collect_logits(model, val_loader, device)
    test_logits, test_targets = collect_logits(model, test_loader, device)

    raw_probs = torch.softmax(test_logits, dim=1).detach().cpu().numpy()
    targets_np = test_targets.detach().cpu().numpy()

    scaler = fit_temperature(val_logits, val_targets, device=device)
    temp_probs = torch.softmax(scaler.temperature_scale(test_logits.to(device)), dim=1).detach().cpu().numpy()

    return raw_probs, temp_probs, targets_np


# =========================================================
# Figure 1: risk-coverage
# =========================================================

def make_risk_coverage_figure(outputs_dir: Path, figs_dir: Path):
    user1_erm = pd.read_csv(outputs_dir / "bigru_erm_user1_train-1_test-1_lp0_0_seed42_selective.csv")
    user1_proto = pd.read_csv(outputs_dir / "bigru_proto001_user1_train-1_test-1_lp0_001_seed42_selective.csv")
    user3_erm = pd.read_csv(outputs_dir / "bigru_erm_user3_train-1_test-1_lp0_0_seed42_selective.csv")
    user3_proto = pd.read_csv(outputs_dir / "bigru_proto001_user3_train-1_test-1_lp0_001_seed42_selective.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, df_erm, df_proto, title in [
        (axes[0], user1_erm, user1_proto, "Held-out user 1"),
        (axes[1], user3_erm, user3_proto, "Held-out user 3"),
    ]:
        ax.plot(df_erm["coverage"], 1.0 - df_erm["selective_acc"], marker="o", label="BiGRU ERM")
        ax.plot(df_proto["coverage"], 1.0 - df_proto["selective_acc"], marker="o", label="BiGRU + Proto(0.001)")
        ax.set_title(title)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out_path = figs_dir / "risk_coverage_users13.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


# =========================================================
# Figure 2: reliability diagrams
# =========================================================

def make_reliability_figure(outputs_dir: Path, figs_dir: Path, metadata_path: str, batch_size: int, num_workers: int, device: str):
    erm_summary = outputs_dir / "bigru_erm_user1_train-1_test-1_lp0_0_seed42_summary.json"
    proto_summary = outputs_dir / "bigru_proto001_user1_train-1_test-1_lp0_001_seed42_summary.json"

    erm_raw, erm_temp, erm_targets = load_run_probs(erm_summary, metadata_path, batch_size, num_workers, device)
    proto_raw, proto_temp, proto_targets = load_run_probs(proto_summary, metadata_path, batch_size, num_workers, device)

    erm_raw_bins = compute_reliability_bins(erm_raw, erm_targets, n_bins=10)
    erm_temp_bins = compute_reliability_bins(erm_temp, erm_targets, n_bins=10)
    proto_raw_bins = compute_reliability_bins(proto_raw, proto_targets, n_bins=10)
    proto_temp_bins = compute_reliability_bins(proto_temp, proto_targets, n_bins=10)

    erm_raw_ece = compute_ece(erm_raw, erm_targets)
    erm_temp_ece = compute_ece(erm_temp, erm_targets)
    proto_raw_ece = compute_ece(proto_raw, proto_targets)
    proto_temp_ece = compute_ece(proto_temp, proto_targets)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    width = 0.08

    # ERM panel
    ax = axes[0]
    centers = erm_raw_bins["bin_centers"]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.bar(centers - width/2, erm_raw_bins["bin_acc"], width=width, alpha=0.7, label=f"Raw (ECE={erm_raw_ece:.3f})")
    ax.bar(centers + width/2, erm_temp_bins["bin_acc"], width=width, alpha=0.7, label=f"Temp. scaled (ECE={erm_temp_ece:.3f})")
    ax.set_title("User 1, BiGRU ERM")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Proto panel
    ax = axes[1]
    centers = proto_raw_bins["bin_centers"]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.bar(centers - width/2, proto_raw_bins["bin_acc"], width=width, alpha=0.7, label=f"Raw (ECE={proto_raw_ece:.3f})")
    ax.bar(centers + width/2, proto_temp_bins["bin_acc"], width=width, alpha=0.7, label=f"Temp. scaled (ECE={proto_temp_ece:.3f})")
    ax.set_title("User 1, BiGRU + Proto(0.001)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = figs_dir / "reliability_temp_scaling_user1.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--figs_dir", type=str, default="figs")
    parser.add_argument("--metadata", type=str, default="datasets/Widardata_npy/metadata_npy.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE = {device}")

    make_risk_coverage_figure(outputs_dir, figs_dir)
    make_reliability_figure(outputs_dir, figs_dir, args.metadata, args.batch_size, args.num_workers, device)


if __name__ == "__main__":
    main()