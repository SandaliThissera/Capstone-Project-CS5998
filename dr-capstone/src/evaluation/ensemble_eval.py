import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, scores_from_probs, screening_metrics

PCOLS = [f"p{i}" for i in range(5)]


def load_probs(runs_dir, runs, fname):
    base = None
    mats = []
    for r in runs:
        df = pd.read_csv(Path(runs_dir) / r / fname)
        assert all(c in df.columns for c in PCOLS), f"{r}: needs a classification run (p0..p4 columns)"
        if base is None:
            base = df[["id_code", "y_true"]]
        else:
            assert (df["id_code"].values == base["id_code"].values).all(), f"{r}: image order differs"
            assert (df["y_true"].values == base["y_true"].values).all(), f"{r}: labels differ"
        mats.append(df[PCOLS].values)
    return base["y_true"].values, np.mean(mats, axis=0)


def operating_points(y, probs, targets):
    """Referable DR (grade >= 2): for each target sensitivity, the highest cut-off on P(grade>=2) that reaches it."""
    pos = y >= 2
    p = probs[:, 2:].sum(1)
    rows = []
    for t in targets:
        best = None
        for th in np.unique(np.round(np.concatenate([p, [0.0]]), 6))[::-1]:
            pred = p >= th
            sens = (pred & pos).sum() / pos.sum()
            if sens >= t:
                spec = (~pred & ~pos).sum() / (~pos).sum()
                best = (float(th), float(sens), float(spec), int((pred & ~pos).sum()), int((~pred & pos).sum()))
                break
        rows.append((t, best))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folder names to average")
    ap.add_argument("--file", default="probs_val_last.csv")
    ap.add_argument("--targets", nargs="*", type=float, default=[0.90, 0.95, 0.97])
    ap.add_argument("--save", default=None, help="write the metrics and operating points to this json")
    a = ap.parse_args()

    y, probs = load_probs(a.runs_dir, a.runs, a.file)
    pred = probs.argmax(1)
    mt = compute_metrics(y, pred)
    mt.update(screening_metrics(y, pred, scores_from_probs(probs)))
    print(f"Ensemble of {len(a.runs)} model(s) on {len(y)} images ({a.file}):")
    print(f"  QWK {mt['qwk']:.3f} | acc {mt['acc']:.3f} | macro-F1 {mt['macro_f1']:.3f}")
    print(f"  recall per grade {mt['recall_per_class']}")
    print(f"  referable DR at argmax: sens {mt['referable_sens']:.3f} | spec {mt['referable_spec']:.3f} | AUC {mt['referable_auc']:.3f}")

    ops = operating_points(y, probs, a.targets)
    print("\nReferable-DR operating points (chosen on this split, so optimistic here; fix the cut-off and apply it unchanged to test):")
    print("  target sens | cut-off on P(grade>=2) | sens   spec   | false alarms  missed")
    for t, o in ops:
        if o is None:
            print(f"  {t:11.2f} | not reachable")
        else:
            print(f"  {t:11.2f} | {o[0]:22.4f} | {o[1]:.3f}  {o[2]:.3f}  | {o[3]:12d}  {o[4]:6d}")
    if a.save:
        Path(a.save).write_text(json.dumps({"runs": a.runs, "metrics": mt,
                                            "operating_points": {str(t): o for t, o in ops}}, indent=2))
        print("saved", a.save)


if __name__ == "__main__":
    main()
