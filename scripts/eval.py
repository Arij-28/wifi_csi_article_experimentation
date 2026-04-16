from __future__ import annotations

import argparse
from pathlib import Path
import torch

from wificsi_exp.utils.config import load_config
from wificsi_exp.utils.seed import set_seed
from wificsi_exp.engine.builder import build_all
from wificsi_exp.engine.evaluator import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    components = build_all(cfg)
    model = components["model"]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    results = evaluate_model(
        model=model,
        loader=components["test_loader"],
        selective_cfg=cfg.get("evaluation", {}),
    )
    print(results)


if __name__ == "__main__":
    main()
