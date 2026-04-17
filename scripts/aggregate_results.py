from pathlib import Path
import json
import pandas as pd

out_dir = Path("outputs")
rows = []

for path in out_dir.glob("*_summary.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = path.stem

    if "erm_proto_large" in name:
        method = "ERM+Proto"
    elif "erm_brier_large" in name:
        method = "ERM+Brier"
    elif "erm_large" in name:
        method = "ERM"
    else:
        continue

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
    })

df = pd.DataFrame(rows)
df.to_csv(out_dir / "all_results.csv", index=False)

summary = (
    df.groupby("method")[["acc", "macro_f1", "ece", "brier", "aurc"]]
      .agg(["mean", "std"])
      .round(4)
)

summary.to_csv(out_dir / "summary_by_method.csv")
print(df)
print("\n=== Summary by method ===")
print(summary)