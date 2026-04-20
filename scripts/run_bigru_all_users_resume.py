from pathlib import Path
import subprocess
import sys
import pandas as pd

METADATA = "datasets/Widardata_npy/metadata_npy.csv"
TRAIN_SCRIPT = "scripts/train_widar_bigru_proto_selective.py"
OUT_DIR = Path("outputs")

SEEDS = [42]
METHODS = [
    ("bigru_erm", 0.0),
    ("bigru_proto001", 0.001),
]

def expected_summary(tag: str, user: int, seed: int, lambda_proto: float) -> Path:
    lp = str(lambda_proto).replace(".", "_")
    return OUT_DIR / f"{tag}_user{user}_train-1_test-1_lp{lp}_seed{seed}_summary.json"

def main():
    df = pd.read_csv(METADATA)
    if "duplicate" in df.columns:
        df = df[df["duplicate"] == False].copy()  # noqa: E712

    users = sorted(df["user"].dropna().astype(int).unique().tolist())
    print("Users found in metadata:", users)

    for tag, lambda_proto in METHODS:
        print(f"\n=== {tag} | lambda_proto={lambda_proto} ===")
        for user in users:
            for seed in SEEDS:
                summary_path = expected_summary(tag, user, seed, lambda_proto)

                if summary_path.exists():
                    print(f"[SKIP] {summary_path.name}")
                    continue

                cmd = [
                    sys.executable,
                    TRAIN_SCRIPT,
                    "--subset_train", "-1",
                    "--subset_test", "-1",
                    "--lambda_proto", str(lambda_proto),
                    "--user_test", str(user),
                    "--epochs", "20",
                    "--seed", str(seed),
                    "--tag", tag,
                ]

                print(f"[RUN ] user={user} seed={seed} tag={tag}")
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()