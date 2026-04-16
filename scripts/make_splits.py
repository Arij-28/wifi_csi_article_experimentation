from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from wificsi_exp.utils.config import load_config
from wificsi_exp.data.splits import make_splits_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--protocol", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    metadata_csv = Path(cfg["dataset"]["metadata_csv"])
    df = pd.read_csv(metadata_csv)
    protocol = args.protocol or cfg["protocol"]["mode"]
    split_df = make_splits_dataframe(df=df, cfg=cfg, protocol_name=protocol)
    out_path = metadata_csv.with_name(metadata_csv.stem + "_with_splits.csv")
    split_df.to_csv(out_path, index=False)
    print(f"Saved splits to {out_path}")


if __name__ == "__main__":
    main()
