from pathlib import Path
import json
import re
import pandas as pd

out_dir = Path("outputs")
rows = []

def extract_user_seed_from_name(name: str):
    user_match = re.search(r"_user(\d+)", name)
    seed_match = re.search(r"_seed(\d+)", name)
    user = int(user_match.group(1)) if user_match else None
    seed = int(seed_match.group(1)) if seed_match else None
    return user, seed

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

    user_from_name, seed_from_name = extract_user_seed_from_name(name)

    best_test = data.get("best_test", {})
    if not best_test:
        continue

    selective_path = path.with_name(path.stem.replace("_summary", "_selective") + ".csv")

    sel50 = None
    sel80 = None
    if selective_path.exists():
        sel_df = pd.read_csv(selective_path)
        if not sel_df.empty:
            sel50 = float(sel_df.iloc[(sel_df["coverage"] - 0.5).abs().argsort()[:1]]["selective_acc"].iloc[0])
            sel80 = float(sel_df.iloc[(sel_df["coverage"] - 0.8).abs().argsort()[:1]]["selective_acc"].iloc[0])

    rows.append({
        "file": name,
        "method": method,
        "user_test": int(data.get("user_test", user_from_name)),
        "seed": int(data.get("seed", seed_from_name)),
        "acc": float(best_test["acc"]),
        "macro_f1": float(best_test["macro_f1"]),
        "ece": float(best_test["ece"]),
        "brier": float(best_test["brier"]),
        "aurc": float(data["aurc"]),
        "sel_acc_50": sel50,
        "sel_acc_80": sel80,
    })

df = pd.DataFrame(rows)

# enlève lignes incomplètes éventuelles
df = df.dropna(subset=["user_test", "seed"]).copy()

# garde un seul run par (method, user_test, seed)
# on préfère les noms propres sans répétition type user3_user3
df["is_clean_name"] = ~df["file"].str.contains(r"_user\d+_user\d+_")
df = (
    df.sort_values(["method", "user_test", "seed", "is_clean_name"], ascending=[True, True, True, False])
      .drop_duplicates(subset=["method", "user_test", "seed"], keep="first")
      .drop(columns=["is_clean_name"])
      .sort_values(["method", "user_test", "seed"])
      .reset_index(drop=True)
)

df.to_csv(out_dir / "bigru_final_all_results_dedup.csv", index=False)

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

summary_method.to_csv(out_dir / "bigru_final_summary_by_method_dedup.csv")
summary_user_method.to_csv(out_dir / "bigru_final_summary_by_user_and_method_dedup.csv")

print("=== Final deduplicated runs ===")
print(df)
print("\n=== Summary by method ===")
print(summary_method)
print("\n=== Summary by user and method ===")
print(summary_user_method)