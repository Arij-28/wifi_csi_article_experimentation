from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
LAMBDA_BRIER = 0.1   # mets 0.1 pour ERM + Brier


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
        return x, y


class Widar_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gru = nn.GRU(400, 64, num_layers=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, 22, 400]
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)  # [22, B, 400]
        _, ht = self.gru(x)
        logits = self.fc(ht[-1])
        return logits


def brier_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = torch.zeros((targets.size(0), num_classes), device=logits.device, dtype=probs.dtype)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return ((probs - onehot) ** 2).sum(dim=1).mean()


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


def run_epoch(model, loader, optimizer=None, lambda_brier: float = 0.0):
    ce_criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        ce = ce_criterion(logits, y)
        brier_reg = brier_loss_from_logits(logits, y, num_classes=logits.size(1))
        loss = ce + lambda_brier * brier_reg

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    probs = torch.softmax(all_logits, dim=1).numpy()
    targets = all_targets.numpy()
    preds = probs.argmax(axis=1)

    metrics = {
        "loss": total_loss / total_n,
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "ece": compute_ece(probs, targets, n_bins=15),
        "brier": compute_brier(probs, targets, num_classes=probs.shape[1]),
    }
    return metrics


@torch.no_grad()
def collect_test_outputs(model, loader):
    model.eval()
    all_logits = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    return probs, preds, conf, targets


def selective_metrics(probs, preds, conf, targets, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)):
    order = np.argsort(-conf)  # confiance décroissante
    probs = probs[order]
    preds = preds[order]
    conf = conf[order]
    targets = targets[order]

    rows = []
    n = len(targets)

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        sel_preds = preds[:k]
        sel_targets = targets[:k]

        sel_acc = float((sel_preds == sel_targets).mean())
        sel_f1 = float(f1_score(sel_targets, sel_preds, average="macro"))
        rows.append({
            "coverage": k / n,
            "num_selected": k,
            "selective_acc": sel_acc,
            "selective_macro_f1": sel_f1,
            "avg_confidence": float(conf[:k].mean()),
        })

    return pd.DataFrame(rows)


def compute_aurc(preds, conf, targets):
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

    aurc = np.trapezoid(risks, coverages)
    return float(aurc)

def main():
    meta_npy = Path("datasets/Widardata_npy/metadata_npy.csv")
    df = pd.read_csv(meta_npy)

    if "duplicate" in df.columns:
        df = df[df["duplicate"] == False].copy()  # noqa: E712

    # LOSO user = 1
    train_df = df[df["user"] != 1].copy()
    test_df = df[df["user"] == 1].copy()

    # subset CPU-friendly
    train_df = train_df.sample(n=min(6000, len(train_df)), random_state=42).reset_index(drop=True)
    test_df = test_df.sample(n=min(2000, len(test_df)), random_state=42).reset_index(drop=True)

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
    print(f"lambda_brier = {LAMBDA_BRIER}")

    best_acc = -1.0
    best_state = None
    best_test = None

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, lambda_brier=LAMBDA_BRIER)
        test_metrics = run_epoch(model, test_loader, optimizer=None, lambda_brier=LAMBDA_BRIER)

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            best_test = test_metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} train_ece={train_metrics['ece']:.4f} "
            f"train_brier={train_metrics['brier']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f} "
            f"test_f1={test_metrics['macro_f1']:.4f} test_ece={test_metrics['ece']:.4f} "
            f"test_brier={test_metrics['brier']:.4f}"
        )

    print("\nBest test metrics:")
    for k, v in best_test.items():
        print(f"{k}: {v:.4f}")

    # recharge le meilleur modèle et calcule la selective prediction
    model.load_state_dict(best_state)
    probs, preds, conf, targets = collect_test_outputs(model, test_loader)

    sel_df = selective_metrics(
        probs, preds, conf, targets,
        coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
    )
    aurc = compute_aurc(preds, conf, targets)

    print("\nSelective prediction results:")
    print(sel_df.to_string(index=False))
    print(f"\nAURC: {aurc:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    tag = f"gru_selective_lambda_{str(LAMBDA_BRIER).replace('.', '_')}"
    sel_df.to_csv(out_dir / f"{tag}.csv", index=False)
    print(f"[OK] selective results saved to {out_dir / f'{tag}.csv'}")


if __name__ == "__main__":
    main()