from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits_dataframe(df: pd.DataFrame, cfg: dict, protocol_name: str) -> pd.DataFrame:
    df = df.copy()
    domain_col = cfg["dataset"]["domain_column"]
    label_col = cfg["dataset"]["label_column"]
    val_ratio = float(cfg["protocol"].get("val_ratio", 0.2))

    if protocol_name == "in_domain":
        if "split" in df.columns:
            return df
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=cfg["seed"], stratify=df[label_col])
        train_only = df.loc[train_idx]
        tr_idx, val_idx = train_test_split(train_only.index, test_size=val_ratio, random_state=cfg["seed"], stratify=train_only[label_col])
        df["split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"
        return df

    if protocol_name == "leave_one_domain_out":
        leave_out = cfg["protocol"]["leave_out_value"]
        df["split"] = "train"
        df.loc[df[domain_col].astype(str) == str(leave_out), "split"] = "test"
        train_only = df[df["split"] == "train"]
        tr_idx, val_idx = train_test_split(train_only.index, test_size=val_ratio, random_state=cfg["seed"], stratify=train_only[label_col])
        df.loc[val_idx, "split"] = "val"
        return df

    raise ValueError(f"Unknown protocol: {protocol_name}")
