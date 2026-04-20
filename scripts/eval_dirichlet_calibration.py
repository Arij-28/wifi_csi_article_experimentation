from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

        x = torch.from_numpy(x).float()   # [22, 400]
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


def subset_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n < 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


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


def compute_brier(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    onehot = np.zeros((len(targets), num_classes), dtype=np.float32)
    onehot[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def evaluate_from_probs(probs: np.ndarray, targets_np: np.ndarray):
    preds = probs.argmax(axis=1)
    return {
        "acc": float((preds == targets_np).mean()),
        "macro_f1": float(f1_score(targets_np, preds, average="macro")),
        "ece": compute_ece(probs, targets_np),
        "brier": compute_brier(probs, targets_np, num_classes=probs.shape[1]),
        "probs": probs,
        "preds": preds,
        "targets": targets_np,
        "conf": probs.max(axis=1),
    }


def selective_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    conf: np.ndarray,
    targets: np.ndarray,
    coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
) -> pd.DataFrame:
    order = np.argsort(-conf)
    preds = preds[order]
    conf = conf[order]
    targets = targets[order]

    rows = []
    n = len(targets)

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        sel_preds = preds[:k]
        sel_targets = targets[:k]

        rows.append({
            "coverage": k / n,
            "num_selected": k,
            "selective_acc": float((sel_preds == sel_targets).mean()),
            "selective_macro_f1": float(f1_score(sel_targets, sel_preds, average="macro")),
            "avg_confidence": float(conf[:k].mean()),
        })

    return pd.DataFrame(rows)


def compute_aurc(preds: np.ndarray, conf: np.ndarray, targets: np.ndarray) -> float:
    order = np.argsort(-conf)
    preds = preds[order]
    targets = targets[order]

    correct = (preds == targets).astype(np.float32)
    risks = []
    coverages = []

    for k in range(1, len(correct) + 1):
        coverages.append(k / len(correct))
        risks.append(1.0 - correct[:k].mean())

    return float(np.trapezoid(risks, coverages))


class DirichletCalibrator(nn.Module):
    """
    Multiclass Dirichlet-style calibration:
    softmax( W log(p) + b )
    where p is the uncalibrated softmax output.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_classes, num_classes, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(num_classes))
            self.linear.bias.zero_()

    def forward_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1).clamp(min=1e-12)
        log_probs = torch.log(probs)
        return self.linear(log_probs)

    @torch.no_grad()
    def predict_proba_from_logits(self, logits: torch.Tensor) -> np.ndarray:
        calibrated_logits = self.forward_from_logits(logits)
        probs = torch.softmax(calibrated_logits, dim=1)
        return probs.detach().cpu().numpy()


def fit_dirichlet_calibrator(
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    device: str,
    lr: float = 0.01,
    max_epochs: int = 500,
    weight_decay: float = 1e-3,
):
    num_classes = val_logits.shape[1]
    calibrator = DirichletCalibrator(num_classes).to(device)

    logits = val_logits.to(device)
    targets = val_targets.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(calibrator.parameters(), lr=lr)

    best_state = None
    best_loss = float("inf")
    patience = 30
    bad_epochs = 0

    for epoch in range(max_epochs):
        calibrator.train()
        optimizer.zero_grad()

        calibrated_logits = calibrator.forward_from_logits(logits)
        ce = criterion(calibrated_logits, targets)

        # Regularize deviation from identity mapping.
        eye = torch.eye(num_classes, device=device)
        reg_w = ((calibrator.linear.weight - eye) ** 2).mean()
        reg_b = (calibrator.linear.bias ** 2).mean()
        loss = ce + weight_decay * (reg_w + reg_b)

        loss.backward()
        optimizer.step()

        current = float(loss.item())
        if current < best_loss - 1e-7:
            best_loss = current
            best_state = {
                "weight": calibrator.linear.weight.detach().cpu().clone(),
                "bias": calibrator.linear.bias.detach().cpu().clone(),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    with torch.no_grad():
        calibrator.linear.weight.copy_(best_state["weight"].to(device))
        calibrator.linear.bias.copy_(best_state["bias"].to(device))

    return calibrator, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=str, required=True, help="Path to *_summary.json from a trained run")
    parser.add_argument("--metadata", type=str, default="datasets/Widardata_npy/metadata_npy.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dc_lr", type=float, default=0.01)
    parser.add_argument("--dc_epochs", type=int, default=500)
    parser.add_argument("--dc_weight_decay", type=float, default=1e-3)
    parser.add_argument("--tag", type=str, default="dirichlet_calibrated")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    user_test = int(summary["user_test"])
    subset_train = int(summary["subset_train"])
    subset_test = int(summary["subset_test"])
    seed = int(summary["seed"])
    val_ratio = float(summary.get("val_ratio", 0.1))

    summary_path = Path(args.summary_json)
    ckpt_path = summary_path.with_name(summary_path.stem.replace("_summary", "_best") + ".pt")

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.metadata)
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

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BiGRUProto(
        num_classes=len(label_to_index),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    print(f"DEVICE = {device}")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"user_test = {user_test}")
    print(f"val size  = {len(val_ds)}")
    print(f"test size = {len(test_ds)}")

    val_logits, val_targets = collect_logits(model, val_loader, device)
    test_logits, test_targets = collect_logits(model, test_loader, device)

    raw_probs = torch.softmax(test_logits, dim=1).detach().cpu().numpy()
    targets_np = test_targets.detach().cpu().numpy()
    raw_test = evaluate_from_probs(raw_probs, targets_np)

    calibrator, best_val_loss = fit_dirichlet_calibrator(
        val_logits=val_logits,
        val_targets=val_targets,
        device=device,
        lr=args.dc_lr,
        max_epochs=args.dc_epochs,
        weight_decay=args.dc_weight_decay,
    )

    dc_probs = calibrator.predict_proba_from_logits(test_logits.to(device))
    dc_test = evaluate_from_probs(dc_probs, targets_np)

    raw_sel = selective_metrics(raw_test["probs"], raw_test["preds"], raw_test["conf"], raw_test["targets"])
    dc_sel = selective_metrics(dc_test["probs"], dc_test["preds"], dc_test["conf"], dc_test["targets"])

    raw_aurc = compute_aurc(raw_test["preds"], raw_test["conf"], raw_test["targets"])
    dc_aurc = compute_aurc(dc_test["preds"], dc_test["conf"], dc_test["targets"])

    print("\n=== Raw ===")
    print(f"acc: {raw_test['acc']:.4f}")
    print(f"macro_f1: {raw_test['macro_f1']:.4f}")
    print(f"ece: {raw_test['ece']:.4f}")
    print(f"brier: {raw_test['brier']:.4f}")
    print(f"aurc: {raw_aurc:.4f}")

    print("\n=== Dirichlet calibrated ===")
    print(f"best_val_loss: {best_val_loss:.6f}")
    print(f"acc: {dc_test['acc']:.4f}")
    print(f"macro_f1: {dc_test['macro_f1']:.4f}")
    print(f"ece: {dc_test['ece']:.4f}")
    print(f"brier: {dc_test['brier']:.4f}")
    print(f"aurc: {dc_aurc:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    run_name = f"{summary_path.stem.replace('_summary', '')}_{args.tag}"

    raw_sel.to_csv(out_dir / f"{run_name}_selective_raw.csv", index=False)
    dc_sel.to_csv(out_dir / f"{run_name}_selective_dirichlet.csv", index=False)

    out = {
        "summary_json": str(summary_path),
        "checkpoint": str(ckpt_path),
        "best_val_loss": best_val_loss,
        "raw": {
            "acc": raw_test["acc"],
            "macro_f1": raw_test["macro_f1"],
            "ece": raw_test["ece"],
            "brier": raw_test["brier"],
            "aurc": raw_aurc,
        },
        "dirichlet": {
            "acc": dc_test["acc"],
            "macro_f1": dc_test["macro_f1"],
            "ece": dc_test["ece"],
            "brier": dc_test["brier"],
            "aurc": dc_aurc,
        },
    }

    with open(out_dir / f"{run_name}_dirichlet_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] saved raw selective to {out_dir / f'{run_name}_selective_raw.csv'}")
    print(f"[OK] saved dirichlet selective to {out_dir / f'{run_name}_selective_dirichlet.csv'}")
    print(f"[OK] saved dirichlet summary to {out_dir / f'{run_name}_dirichlet_summary.json'}")


if __name__ == "__main__":
    main()