from pathlib import Path
import pandas as pd
import json

meta_path = Path("datasets/Widardata/metadata.csv")
out_dir = Path("datasets/Widardata/splits_loso_user")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(meta_path)

# on enlève les doublons si besoin
if "duplicate" in df.columns:
    df = df[df["duplicate"] == False].copy()  # noqa: E712

users = sorted(df["user"].dropna().unique().tolist())

for test_user in users:
    train_df = df[df["user"] != test_user].copy()
    test_df = df[df["user"] == test_user].copy()

    train_path = out_dir / f"train_user_{test_user}.csv"
    test_path = out_dir / f"test_user_{test_user}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] user {test_user}: train={len(train_df)} test={len(test_df)}")

summary = {
    "protocol": "leave-one-user-out",
    "num_folds": len(users),
    "users": users,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[OK] splits sauvegardés dans {out_dir}")