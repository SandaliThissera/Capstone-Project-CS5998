"""Check that the packaged predictor reproduces the stored test probabilities (run once, in Colab)."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.evaluation.gradcam import PCOLS, load_ensemble
from src.inference.predictor import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ref = load_ensemble(a.runs_dir, a.runs, "probs_test_last.csv").sample(a.n, random_state=0)
    pred = Predictor(a.weights, a.device)
    diffs, same = [], 0
    for _, r in ref.iterrows():
        img = cv2.cvtColor(cv2.imread(str(Path(a.cache) / f"{r.id_code}.png")), cv2.COLOR_BGR2RGB)
        out = pred.predict(img, with_cam=False, already_preprocessed=True)
        diffs.append(np.abs(np.array(out["probs"]) - r[PCOLS].values.astype(float)).max())
        same += int(out["grade"] == int(r.y_pred))
    print(f"{a.n} test images | max abs prob difference: mean {np.mean(diffs):.5f}, worst {np.max(diffs):.5f} "
          f"| same predicted grade: {same}/{a.n}")


if __name__ == "__main__":
    main()
