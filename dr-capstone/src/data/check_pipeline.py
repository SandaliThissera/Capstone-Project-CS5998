import argparse
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset, build_transforms, class_weights
from src.data.preprocess import ben_graham


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="configs/splits_aptos.csv")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="reports/figures/preprocessing_before_after.png")
    a = ap.parse_args()

    df = pd.read_csv(a.splits)
    ids = {s: set(df[df.split == s].id_code) for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    print("split overlap: none |", {s: len(v) for s, v in ids.items()})
    print("class weights (train):", class_weights(df[df.split == "train"]).numpy().round(3))

    ds = DRDataset(a.splits, "train", a.cache, build_transforms(True, a.size))
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    t0 = time.time()
    xb, yb = next(iter(dl))
    print("batch:", tuple(xb.shape), xb.dtype, f"min={xb.min().item():.2f} max={xb.max().item():.2f}",
          "labels:", yb.tolist(), f"({time.time() - t0:.1f}s)")

    sample = df[df.split == "train"].sample(6, random_state=3)
    fig, axes = plt.subplots(6, 3, figsize=(9, 18))
    for r, (_, row) in enumerate(sample.iterrows()):
        raw = read_rgb(Path(a.raw) / f"{row.id_code}.png")
        raw = cv2.resize(raw, (int(raw.shape[1] * 400 / raw.shape[0]), 400))
        proc = read_rgb(Path(a.cache) / f"{row.id_code}.png")
        titles = ("original", "cropped + square + resized", "Ben Graham enhanced")
        for ax, im, t in zip(axes[r], (raw, proc, ben_graham(proc)), titles):
            ax.imshow(im); ax.axis("off"); ax.set_title(f"Grade {row.diagnosis} - {t}", fontsize=8)
    plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=70)
    print("saved", a.out)


if __name__ == "__main__":
    main()
