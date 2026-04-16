from __future__ import annotations

import argparse
import pandas as pd

from wificsi_exp.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    meta = pd.read_csv(cfg["dataset"]["metadata_csv"])
    print("Rows:", len(meta))
    print("Columns:", list(meta.columns))
    for col in [cfg["dataset"]["label_column"], cfg["dataset"]["domain_column"]]:
        if col in meta.columns:
            print(f"Unique {col}:", meta[col].nunique())
            print(meta[col].value_counts().head())


if __name__ == "__main__":
    main()
