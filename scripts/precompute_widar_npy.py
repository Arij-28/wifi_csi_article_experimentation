from pathlib import Path
import numpy as np
import pandas as pd


def main():
    meta_path = Path("datasets/Widardata/metadata.csv")
    out_root = Path("datasets/Widardata_npy")
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_path)

    # optionnel: retirer les doublons
    if "duplicate" in df.columns:
        df = df[df["duplicate"] == False].copy()  # noqa: E712

    new_paths = []

    for i, row in df.iterrows():
        csv_path = Path(row["path"])
        rel_path = Path(row["relative_path"]).with_suffix(".npy")
        npy_path = out_root / rel_path
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        x = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        np.save(npy_path, x)
        new_paths.append(str(npy_path.resolve()))

        if (i + 1) % 1000 == 0:
            print(f"[INFO] {i+1}/{len(df)} fichiers convertis")

    df["npy_path"] = new_paths

    out_meta = out_root / "metadata_npy.csv"
    df.to_csv(out_meta, index=False, encoding="utf-8")

    print(f"[OK] Conversion terminée")
    print(f"[OK] NPY root: {out_root}")
    print(f"[OK] Metadata: {out_meta}")
    print(f"[OK] Total fichiers: {len(df)}")


if __name__ == "__main__":
    main()