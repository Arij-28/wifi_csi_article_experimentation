from __future__ import annotations

import argparse
from pathlib import Path

from wificsi_exp.utils.config import load_config
from wificsi_exp.utils.seed import set_seed
from wificsi_exp.engine.builder import build_all
from wificsi_exp.engine.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="default_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    components = build_all(cfg)
    trainer = Trainer(
        model=components["model"],
        optimizer=components["optimizer"],
        train_loader=components["train_loader"],
        val_loader=components["val_loader"],
        cfg=cfg,
        class_names=components["class_names"],
    )
    trainer.fit(run_name=args.run_name)


if __name__ == "__main__":
    main()
