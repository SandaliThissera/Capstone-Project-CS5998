import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.data.preprocess import preprocess


def _work(job):
    src, dst, size = job
    if dst.exists():
        return 0
    img = cv2.imread(str(src))
    if img is None:
        return -1
    img = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)
    cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="configs/splits_aptos.csv")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(a.splits)["id_code"]
    jobs = [(Path(a.raw) / f"{i}.png", out / f"{i}.png", a.size) for i in ids]
    with Pool(a.workers) as p:
        res = list(tqdm(p.imap_unordered(_work, jobs, chunksize=8), total=len(jobs)))
    print("written", res.count(1), "| skipped", res.count(0), "| failed", res.count(-1))


if __name__ == "__main__":
    main()
