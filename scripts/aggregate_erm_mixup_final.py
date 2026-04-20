from pathlib import Path
import json
import re
import pandas as pd

OUT_DIR = Path("outputs")

ERM_GLOB = "bigru_erm_user*_train-1_test-1_lp0_0_seed*_summary.json"
MIXUP_GLOB = "bigru_mixup_user*_train-1_test-1_ma0_4_mp0_5_seed*_summary.json"


def infer_user_from_name(name: str):
    m = re.search(r"_user(\d+)_", name)
    return int(m.group(1)) if m else None


def infer_seed_from_name(name: str):
    m = re.search(r"_seed(\d+)", name)
    return int(m.group(1)) if m else None


def load_run(summary_path: Path, method_name: str):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_test = data.get("best_test", {})

    user_test = data.get("user_test", infer_user_from_name(summary_path.name))
    seed = data.get("seed", infer_seed_from_name(summary_path.name))
    aurc = data.get("aurc", None)

    selective_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_selective.csv"))
    sel_acc_50 = None
    sel_acc_80 = None

    if selective_path.exists():
        sel_df = pd.read_csv(selective_path)

        if "coverage" in sel_df.columns and "selective_acc" in sel_df.columns:
            row_50 = sel_df.iloc[(sel_df["coverage"] - 0.5).abs().argsort()[:1]]
            row_80 = sel_df.iloc[(sel_df["coverage"] - 0.8).abs().argsort()[:1]]

            if len(row_50) > 0:
                sel_acc_50 = float(row_50["selective_acc"].iloc[0])
            if len(row_80) > 0:
                sel_acc_80 = float(row_80["selective_acc"].iloc[0])

    return {
        "file": summary_path.name,
        "method": method_name,
        "user_test": int(user_test) if user_test is not None else None,
        "seed": int(seed) if seed is not None else None,
        "acc": float(best_test.get("acc")) if "acc" in best_test else None,
        "macro_f1": float(best_test.get("macro_f1")) if "macro_f1" in best_test else None,
        "ece": float(best_test.get("ece")) if "ece" in best_test else None,
        "brier": float(best_test.get("brier")) if "brier" in best_test else None,
        "aurc": float(aurc) if aurc is not None else None,
        "sel_acc_50": sel_acc_50,
        "sel_acc_80": sel_acc_80,
    }


def collect_runs():
    rows = []

    for path in sorted(OUT_DIR.glob(ERM_GLOB)):
        # skip temp-scaled summaries if any weird names match in future
        if "_temp_" in path.name or "temp_scaled" in path.name:
            continue
        rows.append(load_run(path, "BiGRU ERM"))

    for path in sorted(OUT_DIR.glob(MIXUP_GLOB)):
        if "_temp_" in path.name or "temp_scaled" in path.name:
            continue
        rows.append(load_run(path, "BiGRU + Mixup"))

    df = pd.DataFrame(rows)

    # keep only valid rows
    df = df.dropna(subset=["user_test", "seed", "acc", "macro_f1", "ece", "brier", "aurc"]).copy()

    # keep only users 1,3,5 and seeds 42,123,999 for the paper table
    df = df[df["user_test"].isin([1, 3, 5])].copy()
    df = df[df["seed"].isin([42, 123, 999])].copy()

    # drop duplicate files if any
    df = df.drop_duplicates(subset=["method", "user_test", "seed"], keep="first").copy()

    return df


def main():
    df = collect_runs()

    if df.empty:
        print("No matching runs found.")
        return

    df = df.sort_values(["method", "user_test", "seed"]).reset_index(drop=True)

    print("\n=== Final runs: ERM vs Mixup ===")
    print(df.to_string(index=False))

    summary_by_method = df.groupby("method")[[
        "acc", "macro_f1", "ece", "brier", "aurc", "sel_acc_50", "sel_acc_80"
    ]].agg(["mean", "std"])

    print("\n=== Summary by method ===")
    print(summary_by_method)

    summary_by_user = df.groupby(["method", "user_test"])[[
        "acc", "macro_f1", "ece", "brier", "aurc", "sel_acc_50", "sel_acc_80"
    ]].agg(["mean", "std"])

    print("\n=== Summary by user and method ===")
    print(summary_by_user)

    # save
    df.to_csv(OUT_DIR / "erm_vs_mixup_all_runs.csv", index=False)
    summary_by_method.to_csv(OUT_DIR / "erm_vs_mixup_summary_by_method.csv")
    summary_by_user.to_csv(OUT_DIR / "erm_vs_mixup_summary_by_user.csv")

    print(f"\n[OK] saved {OUT_DIR / 'erm_vs_mixup_all_runs.csv'}")
    print(f"[OK] saved {OUT_DIR / 'erm_vs_mixup_summary_by_method.csv'}")
    print(f"[OK] saved {OUT_DIR / 'erm_vs_mixup_summary_by_user.csv'}")


if __name__ == "__main__":
    main()