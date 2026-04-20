from pathlib import Path
import re
import subprocess
import sys

OUT_DIR = Path("outputs")
TRAIN_SCRIPT = "scripts/train_widar_bigru_proto_selective.py"
SEED = 42
TAG_ERM = "bigru_erm"
TAG_PROTO = "bigru_proto001"
LAMBDA_PROTO = 0.001

def extract_user(path: Path):
    m = re.search(r"_user(\d+)_", path.stem)
    return int(m.group(1)) if m else None

def main():
    # users déjà faits avec ERM, seed 42
    erm_files = list(OUT_DIR.glob(f"{TAG_ERM}_user*_train-1_test-1_lp0_0_seed{SEED}_summary.json"))
    users_done = sorted({extract_user(p) for p in erm_files if extract_user(p) is not None})

    print("Users ERM déjà disponibles:", users_done)

    for user in users_done:
        proto_summary = OUT_DIR / f"{TAG_PROTO}_user{user}_train-1_test-1_lp0_001_seed{SEED}_summary.json"

        if proto_summary.exists():
            print(f"[SKIP] Proto déjà fait pour user {user}")
            continue

        cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--subset_train", "-1",
            "--subset_test", "-1",
            "--lambda_proto", str(LAMBDA_PROTO),
            "--user_test", str(user),
            "--epochs", "20",
            "--seed", str(SEED),
            "--tag", TAG_PROTO,
        ]

        print(f"[RUN ] Proto user={user}")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()