import argparse, hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def make_splits(csv_path, images_dir, out_path, seed=42, val_size=0.15, test_size=0.15):
    df = pd.read_csv(csv_path).drop_duplicates(subset="id_code")
    df["hash"] = [file_md5(Path(images_dir) / f"{i}.png") for i in df["id_code"]]

    n0 = len(df)
    conflict = df.groupby("hash")["diagnosis"].transform("nunique") > 1
    df = df[~conflict].drop_duplicates(subset="hash")
    print(f"Removed {n0 - len(df)} duplicate/conflicting images ({int(conflict.sum())} had conflicting labels)")
    df = df.drop(columns="hash")

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["diagnosis"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val["diagnosis"], random_state=seed)

    out = pd.concat([train.assign(split="train"),
                     val.assign(split="val"),
                     test.assign(split="test")]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.groupby(["split", "diagnosis"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="configs/splits_aptos.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_splits(a.csv, a.images, a.out, a.seed)
