from pathlib import Path
import argparse
import json
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BiGRUSupCon(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=400,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        feat_dim = hidden_dim * 2
        self.fc = nn.Linear(feat_dim, num_classes)

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        # x: [B, 22, 400]
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)  # [22, B, 400]

        _, ht = self.gru(x)
        feat_fwd = ht[-2]
        feat_bwd = ht[-1]
        feat = torch.cat([feat_fwd, feat_bwd], dim=1)

        logits = self.fc(feat)
        proj = self.proj(feat)
        proj = F.normalize(proj, dim=1)

        return logits, feat, proj


def supervised_contrastive_cross_user_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    temperature: float = 0.1,
    cross_user_weight: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Supervised contrastive loss on a single view per sample.
    Positives = same class.
    Cross-user positives receive larger weight.
    """

    device = z.device
    batch_size = z.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    labels = labels.contiguous().view(-1, 1)
    domains = domains.contiguous().view(-1, 1)

    same_label = torch.eq(labels, labels.T).float().to(device)
    same_domain = torch.eq(domains, domains.T).float().to(device)

    eye = torch.eye(batch_size, device=device)
    non_self = 1.0 - eye

    pos_mask = same_label * non_self

    # Weight positives more when they come from different users.
    pos_weights = pos_mask * (same_domain + (1.0 - same_domain) * cross_user_weight)

    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * non_self
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    pos_weight_sum = pos_weights.sum(dim=1)
    valid = pos_weight_sum > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (pos_weights * log_prob).sum(dim=1) / (pos_weight_sum + eps)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


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


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    lambda_supcon: float = 0.0,
    supcon_temperature: float = 0.1,
    cross_user_weight: float = 2.0,
):
    criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_n = 0
    all_logits = []
    all_targets = []

    for x, y, domain in loader:
        x = x.to(device)
        y = y.to(device)
        domain = domain.to(device)

        logits, _, proj = model(x)
        ce = criterion(logits, y)
        supcon = supervised_contrastive_cross_user_loss(
            z=proj,
            labels=y,
            domains=domain,
            temperature=supcon_temperature,
            cross_user_weight=cross_user_weight,
        )

        loss = ce + lambda_supcon * supcon

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_supcon += supcon.item() * bs
        total_n += bs

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    probs = torch.softmax(all_logits, dim=1).numpy()
    targets = all_targets.numpy()
    preds = probs.argmax(axis=1)

    return {
        "loss": total_loss / total_n,
        "ce": total_ce / total_n,
        "supcon": total_supcon / total_n,
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "ece": compute_ece(probs, targets),
        "brier": compute_brier(probs, targets, num_classes=probs.shape[1]),
    }


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for x, y, _ in loader:
        x = x.to(device)
        logits, _, _ = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return probs, preds, conf, targets


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
        coverage = k / len(correct)
        risk = 1.0 - correct[:k].mean()
        coverages.append(coverage)
        risks.append(risk)

    return float(np.trapezoid(risks, coverages))


def subset_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n < 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="datasets/Widardata_npy/metadata_npy.csv")
    parser.add_argument("--user_test", type=int, default=1)
    parser.add_argument("--subset_train", type=int, default=-1)
    parser.add_argument("--subset_test", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_supcon", type=float, default=0.1)
    parser.add_argument("--supcon_temperature", type=float, default=0.1)
    parser.add_argument("--cross_user_weight", type=float, default=2.0)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tag", type=str, default="bigru_supcon")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.metadata)
    if "duplicate" in df.columns:
        df = df[df["duplicate"] == False].copy()

    source_df = df[df["user"] != args.user_test].copy()
    test_df = df[df["user"] == args.user_test].copy()

    source_df = subset_df(source_df, args.subset_train, args.seed)
    test_df = subset_df(test_df, args.subset_test, args.seed)

    train_df, val_df = train_test_split(
        source_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=source_df["label_id"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    labels = sorted(train_df["label_id"].dropna().astype(int).unique().tolist())
    label_to_index = {lab: i for i, lab in enumerate(labels)}

    train_ds = WidarNpyDataset(train_df, label_to_index=label_to_index, normalize=True)
    val_ds = WidarNpyDataset(val_df, label_to_index=label_to_index, normalize=True)
    test_ds = WidarNpyDataset(test_df, label_to_index=label_to_index, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BiGRUSupCon(
        num_classes=len(label_to_index),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"DEVICE = {device}")
    print(f"user_test = {args.user_test}")
    print(f"train size = {len(train_ds)}")
    print(f"val size   = {len(val_ds)}")
    print(f"test size  = {len(test_ds)}")
    print(f"num classes = {len(label_to_index)}")
    print(f"lambda_supcon = {args.lambda_supcon}")
    print(f"supcon_temperature = {args.supcon_temperature}")
    print(f"cross_user_weight = {args.cross_user_weight}")
    print(f"batch_size = {args.batch_size}")

    history = []
    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            lambda_supcon=args.lambda_supcon,
            supcon_temperature=args.supcon_temperature,
            cross_user_weight=args.cross_user_weight,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            lambda_supcon=args.lambda_supcon,
            supcon_temperature=args.supcon_temperature,
            cross_user_weight=args.cross_user_weight,
        )

        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            optimizer=None,
            lambda_supcon=args.lambda_supcon,
            supcon_temperature=args.supcon_temperature,
            cross_user_weight=args.cross_user_weight,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_metrics["ece"],
            "test_brier": test_metrics["brier"],
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['macro_f1']:.4f} | "
            f"test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['macro_f1']:.4f} "
            f"test_ece={test_metrics['ece']:.4f} test_brier={test_metrics['brier']:.4f}"
        )

    model.load_state_dict(best_state)
    best_test = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        lambda_supcon=args.lambda_supcon,
        supcon_temperature=args.supcon_temperature,
        cross_user_weight=args.cross_user_weight,
    )

    print("\nBest model selected on val_macro_f1")
    print(f"best_epoch: {best_epoch}")
    for k, v in best_test.items():
        print(f"{k}: {v:.4f}")

    probs, preds, conf, targets = collect_outputs(model, test_loader, device)
    sel_df = selective_metrics(probs, preds, conf, targets)
    aurc = compute_aurc(preds, conf, targets)

    print("\nSelective prediction results:")
    print(sel_df.to_string(index=False))
    print(f"\nAURC: {aurc:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    run_name = (
        f"{args.tag}"
        f"_user{args.user_test}"
        f"_train{args.subset_train}"
        f"_test{args.subset_test}"
        f"_lsc{str(args.lambda_supcon).replace('.', '_')}"
        f"_tau{str(args.supcon_temperature).replace('.', '_')}"
        f"_cw{str(args.cross_user_weight).replace('.', '_')}"
        f"_seed{args.seed}"
    )

    pd.DataFrame(history).to_csv(out_dir / f"{run_name}_history.csv", index=False)
    sel_df.to_csv(out_dir / f"{run_name}_selective.csv", index=False)

    summary = {
        "device": device,
        "user_test": args.user_test,
        "subset_train": args.subset_train,
        "subset_test": args.subset_test,
        "val_ratio": args.val_ratio,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_supcon": args.lambda_supcon,
        "supcon_temperature": args.supcon_temperature,
        "cross_user_weight": args.cross_user_weight,
        "proj_dim": args.proj_dim,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_test": best_test,
        "aurc": aurc,
    }

    with open(out_dir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(best_state, out_dir / f"{run_name}_best.pt")
    print(f"\n[OK] saved history to {out_dir / f'{run_name}_history.csv'}")
    print(f"[OK] saved selective results to {out_dir / f'{run_name}_selective.csv'}")
    print(f"[OK] saved summary to {out_dir / f'{run_name}_summary.json'}")
    print(f"[OK] saved weights to {out_dir / f'{run_name}_best.pt'}")


if __name__ == "__main__":
    main()