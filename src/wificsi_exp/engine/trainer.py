from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from wificsi_exp.losses.prototype import prototype_consistency_loss
from wificsi_exp.losses.supcon import supervised_contrastive_loss
from wificsi_exp.losses.brier import brier_loss
from wificsi_exp.engine.evaluator import evaluate_model
from wificsi_exp.utils.io import ensure_dir, save_json


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, cfg: dict, class_names: list[str]):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_metric = -1.0
        self.best_prototypes = {}

    def _compute_loss(self, batch):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        domains = batch["domain"]
        out = self.model(x)
        logits = out["logits"]
        embeddings = out["embeddings"]

        loss_cls = F.cross_entropy(logits, y)
        objective = self.cfg["training"]["objective"]
        loss = loss_cls
        proto_bank = {}

        if "proto" in objective:
            loss_proto, proto_bank = prototype_consistency_loss(embeddings, y, domains)
            loss = loss + self.cfg["training"]["lambda_proto"] * loss_proto
        if "supcon" in objective:
            loss_supcon = supervised_contrastive_loss(embeddings, y, self.cfg["training"]["temperature"])
            loss = loss + self.cfg["training"]["lambda_supcon"] * loss_supcon
        if "brier" in objective:
            loss_b = brier_loss(logits, y, num_classes=self.cfg["model"]["num_classes"])
            loss = loss + self.cfg["training"]["lambda_brier"] * loss_b
        return loss, proto_bank

    def fit(self, run_name: str) -> None:
        run_dir = ensure_dir(Path(self.cfg["output_dir"]) / run_name)
        best_path = run_dir / "best.pt"
        patience = 0
        max_patience = int(self.cfg["training"]["early_stop_patience"])

        for epoch in range(1, int(self.cfg["training"]["epochs"]) + 1):
            self.model.train()
            epoch_loss = 0.0
            last_proto = {}
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
                self.optimizer.zero_grad()
                loss, proto_bank = self._compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())
                if proto_bank:
                    last_proto = proto_bank

            val_metrics = evaluate_model(self.model, self.val_loader, selective_cfg=self.cfg.get("evaluation", {}), prototype_bank=last_proto)
            score = val_metrics["macro_f1"]
            print({"epoch": epoch, "train_loss": epoch_loss / max(1, len(self.train_loader)), **val_metrics})
            if score > self.best_metric:
                self.best_metric = score
                self.best_prototypes = last_proto
                torch.save({"model_state": self.model.state_dict(), "epoch": epoch}, best_path)
                save_json(val_metrics, run_dir / "best_val_metrics.json")
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
