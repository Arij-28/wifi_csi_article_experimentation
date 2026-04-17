from pathlib import Path
import json
import pandas as pd

out_dir = Path("outputs")
rows = []

for path in out_dir.glob("*_summary.json"):
    name = path.stem

    if name.startswith("bigru_erm_user"):
        method = "BiGRU ERM"
    elif name.startswith("bigru_proto001_user"):
        method = "BiGRU+Proto(0.001)"
    else:
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    selective_path = path.with_name(path.stem.replace("_summary", "_selective") + ".csv")

    sel50 = None
    sel80 = None
    if selective_path.exists():
        sel_df = pd.read_csv(selective_path)
        sel50 = float(sel_df.iloc[(sel_df["coverage"] - 0.5).abs().argsort()[:1]]["selective_acc"].iloc[0])
        sel80 = float(sel_df.iloc[(sel_df["coverage"] - 0.8).abs().argsort()[:1]]["selective_acc"].iloc[0])

    rows.append({
        "file": name,
        "method": method,
        "user_test": data["user_test"],
        "seed": data["seed"],
        "acc": data["best_test"]["acc"],
        "macro_f1": data["best_test"]["macro_f1"],
        "ece": data["best_test"]["ece"],
        "brier": data["best_test"]["brier"],
        "aurc": data["aurc"],
        "sel_acc_50": sel50,
        "sel_acc_80": sel80,
    })

df = pd.DataFrame(rows).sort_values(["method", "user_test", "seed"])
df.to_csv(out_dir / "bigru_final_all_results.csv", index=False)

summary_method = (
    df.groupby("method")[["acc", "macro_f1", "ece", "brier", "aurc", "sel_acc_50", "sel_acc_80"]]
      .agg(["mean", "std"])
      .round(4)
)

summary_user_method = (
    df.groupby(["method", "user_test"])[["acc", "macro_f1", "ece", "brier", "aurc", "sel_acc_50", "sel_acc_80"]]
      .agg(["mean", "std"])
      .round(4)
)

summary_method.to_csv(out_dir / "bigru_final_summary_by_method.csv")
summary_user_method.to_csv(out_dir / "bigru_final_summary_by_user_and_method.csv")

print("=== All final runs ===")
print(df)
print("\n=== Summary by method ===")
print(summary_method)
print("\n=== Summary by user and method ===")
print(summary_user_method)