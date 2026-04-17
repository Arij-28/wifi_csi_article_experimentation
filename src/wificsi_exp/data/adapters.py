from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Helpers
# =========================================================

def split_label_dir(dirname: str) -> Tuple[Optional[int], str]:
    """
    Ex:
        '1-Push&Pull' -> (1, 'Push&Pull')
        '10-Draw-Zigzag(V)' -> (10, 'Draw-Zigzag(V)')
    """
    if "-" not in dirname:
        return None, dirname

    left, right = dirname.split("-", 1)
    try:
        return int(left), right
    except ValueError:
        return None, dirname


def clean_duplicate_suffix(stem: str) -> Tuple[str, bool]:
    """
    Ex:
        abc(1) -> (abc, True)
        abc    -> (abc, False)
    """
    cleaned = re.sub(r"\(\d+\)$", "", stem)
    return cleaned, cleaned != stem


def parse_filename(filename: str) -> Dict[str, object]:
    """
    Parse un nom du type :
        user1-1-1-1-1-1-1e-07-100-20-100000-L0.csv

    Structure attendue après split('-') :
        [0]  user1
        [1]  f1
        [2]  f2
        [3]  f3
        [4]  f4
        [5]  f5
        [6]  1e
        [7]  07
        [8]  100
        [9]  20
        [10] 100000
        [11] L0
    """
    stem = Path(filename).stem
    clean_stem, is_duplicate = clean_duplicate_suffix(stem)
    parts = clean_stem.split("-")

    if len(parts) != 12:
        raise ValueError(f"Nom inattendu ({len(parts)} segments): {filename}")

    user_token = parts[0]
    if not user_token.startswith("user"):
        raise ValueError(f"Préfixe invalide: {filename}")

    try:
        user = int(user_token.replace("user", ""))
        f1 = int(parts[1])
        f2 = int(parts[2])
        f3 = int(parts[3])
        f4 = int(parts[4])
        f5 = int(parts[5])
    except Exception as e:
        raise ValueError(f"Erreur parsing champs entiers: {filename} | {e}")

    param1 = parts[6] + "-" + parts[7]   # reconstruit 1e-07
    param2 = parts[8]
    param3 = parts[9]
    param4 = parts[10]
    level = parts[11]

    return {
        "user": user,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "param1": param1,
        "param2": param2,
        "param3": param3,
        "param4": param4,
        "level": level,
        "duplicate": is_duplicate,
    }


# =========================================================
# Metadata builder
# =========================================================

def build_widar_metadata(
    data_root: str | Path,
    save: bool = True,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    root = Path(data_root).resolve()
    print(f"[INFO] data_root = {root}")

    if not root.exists():
        raise FileNotFoundError(f"Dossier introuvable: {root}")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    for split in ["train", "test"]:
        split_dir = root / split
        print(f"[INFO] scan split: {split_dir}")

        if not split_dir.exists():
            print(f"[WARN] split absent: {split_dir}")
            continue

        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        print(f"[INFO] {split}: {len(class_dirs)} dossiers de classes détectés")

        for class_dir in sorted(class_dirs):
            label_id, label_name = split_label_dir(class_dir.name)
            csv_files = sorted(class_dir.glob("*.csv"))
            print(f"  - {class_dir.name}: {len(csv_files)} fichiers")

            for csv_path in csv_files:
                try:
                    meta = parse_filename(csv_path.name)
                except Exception as e:
                    skipped.append(f"{csv_path.name} -> {e}")
                    continue

                clean_filename_stem, _ = clean_duplicate_suffix(csv_path.stem)
                clean_filename = clean_filename_stem + csv_path.suffix

                row = {
                    "path": str(csv_path.resolve()),
                    "relative_path": str(csv_path.relative_to(root).as_posix()),
                    "split": split,
                    "label_id": label_id,
                    "label_name": label_name,
                    "filename": csv_path.name,
                    "clean_filename": clean_filename,
                    **meta,
                }
                rows.append(row)

    if len(rows) == 0:
        print("[DEBUG] Aucun fichier parsé correctement.")
        print("[DEBUG] Exemples d'erreurs:")
        for s in skipped[:20]:
            print("   ", s)
        raise RuntimeError(
            "Aucune ligne construite. Vérifie la structure "
            "'datasets/Widardata/train/<classe>/*.csv' et 'test/<classe>/*.csv'."
        )

    df = pd.DataFrame(rows)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(
            subset=["split", "label_id", "clean_filename"]
        ).reset_index(drop=True)
        after = len(df)
        print(f"[INFO] doublons retirés: {before - after}")

    df = df.sort_values(
        ["split", "label_id", "user", "filename"]
    ).reset_index(drop=True)

    if save:
        out_path = root / "metadata.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] metadata sauvegardé dans: {out_path}")

    print(f"[INFO] shape metadata: {df.shape}")

    if skipped:
        print(f"[WARN] fichiers ignorés: {len(skipped)}")
        for s in skipped[:20]:
            print("   ", s)

    return df


# =========================================================
# CSV inspection
# =========================================================

def inspect_one_csv(csv_path: str | Path) -> None:
    p = Path(csv_path)
    print(f"[INFO] lecture: {p}")

    x = np.loadtxt(p, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    print(f"[OK] shape = {x.shape}")
    print(f"[OK] dtype = {x.dtype}")
    print(f"[OK] nan = {np.isnan(x).any()}")
    print(f"[OK] min = {x.min()}")
    print(f"[OK] max = {x.max()}")


# =========================================================
# Simple dataset wrapper
# =========================================================

class WidarCsvDataset:
    def __init__(
        self,
        data_root: str | Path,
        split: Optional[str] = None,
        metadata_csv: Optional[str | Path] = None,
        keep_duplicates: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()

        if metadata_csv is None:
            metadata_csv = self.data_root / "metadata.csv"
            if not Path(metadata_csv).exists():
                print("[INFO] metadata.csv absent -> génération automatique")
                build_widar_metadata(
                    self.data_root,
                    save=True,
                    drop_duplicates=not keep_duplicates,
                )

        self.df = pd.read_csv(metadata_csv)

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        if not keep_duplicates and "duplicate" in self.df.columns:
            self.df = self.df[self.df["duplicate"] == False].copy()  # noqa: E712

        self.df = self.df.reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("Dataset vide après filtrage.")

        labels = sorted(self.df["label_id"].dropna().astype(int).unique().tolist())
        self.label_to_index = {lab: i for i, lab in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.loadtxt(row["path"], delimiter=",", dtype=np.float32)

        if x.ndim == 1:
            x = x[None, :]

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.label_to_index[int(row["label_id"])]
        meta = row.to_dict()
        return x, y, meta


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/Widardata",
        help="Chemin vers Widardata",
    )
    parser.add_argument(
        "--build-metadata",
        action="store_true",
        help="Construire metadata.csv",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Retirer les doublons (1), (2), ...",
    )
    parser.add_argument(
        "--inspect-first",
        action="store_true",
        help="Lire le premier CSV trouvé et afficher sa shape",
    )
    parser.add_argument(
        "--test-dataset",
        action="store_true",
        help="Tester le wrapper dataset",
    )
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    print(f"[INFO] root = {root}")

    if args.build_metadata:
        df = build_widar_metadata(
            data_root=root,
            save=True,
            drop_duplicates=args.drop_duplicates,
        )
        print(df.head())

    if args.inspect_first:
        first_csv = None
        for split in ["train", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    files = sorted(class_dir.glob("*.csv"))
                    if files:
                        first_csv = files[0]
                        break
            if first_csv is not None:
                break

        if first_csv is None:
            raise RuntimeError("Aucun CSV trouvé pour inspection.")
        inspect_one_csv(first_csv)

    if args.test_dataset:
        ds = WidarCsvDataset(root, split="train", keep_duplicates=False)
        print(f"[OK] dataset train length = {len(ds)}")
        x, y, meta = ds[0]
        print(f"[OK] sample shape = {x.shape}")
        print(f"[OK] y = {y}")
        print(f"[OK] label_name = {meta['label_name']}")
        print(f"[OK] user = {meta['user']}")


if __name__ == "__main__":
    main()