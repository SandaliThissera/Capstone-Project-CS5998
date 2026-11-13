import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold

DEFAULT = (0.5, 1.5, 2.5, 3.5)


def qwk(y, pred):
    return cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(5)))


def decode(score, th):
    return np.digitize(score, np.sort(th))


def fit_thresholds(score, y, start=DEFAULT, sweeps=4, width=1.0, steps=41):
    """Coordinate ascent: move one cut-point at a time to maximise QWK."""
    th = np.array(start, float)
    best = qwk(y, decode(score, th))
    for _ in range(sweeps):
        for i in range(4):
            lo = th[i - 1] + 1e-3 if i > 0 else th[i] - width
            hi = th[i + 1] - 1e-3 if i < 3 else th[i] + width
            for c in np.linspace(max(lo, th[i] - width), min(hi, th[i] + width), steps):
                trial = th.copy(); trial[i] = c
                q = qwk(y, decode(score, trial))
                if q > best + 1e-9:
                    best, th = q, trial
    return th, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="probs_val_last.csv of a regression run")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--save", default=None, help="write thresholds fitted on all of val to this json")
    a = ap.parse_args()

    df = pd.read_csv(a.probs)
    assert "score" in df.columns, "this needs a regression run (a 'score' column)"
    y, s = df["y_true"].values, df["score"].values

    default_pred = decode(s, DEFAULT)
    th_all, q_in = fit_thresholds(s, y)
    res = {"default": (qwk(y, default_pred), (default_pred == y).mean(), f1_score(y, default_pred, average="macro")),
           "tuned_in_sample (optimistic)": (q_in, (decode(s, th_all) == y).mean(), f1_score(y, decode(s, th_all), average="macro"))}

    cv = []
    for r in range(a.repeats):
        oof = np.zeros_like(y)
        for tr, te in StratifiedKFold(a.folds, shuffle=True, random_state=r).split(s, y):
            th, _ = fit_thresholds(s[tr], y[tr])
            oof[te] = decode(s[te], th)
        cv.append((qwk(y, oof), (oof == y).mean(), f1_score(y, oof, average="macro")))
    res["tuned_cross_validated (honest)"] = tuple(np.mean(cv, axis=0))

    print(f"{'decoding':34s} {'QWK':>6s} {'acc':>6s} {'macroF1':>8s}")
    for k, (q, ac, f1) in res.items():
        print(f"{k:34s} {q:6.3f} {ac:6.3f} {f1:8.3f}")
    print("thresholds fitted on all val:", np.round(th_all, 3).tolist())
    print(f"cross-validated QWK spread over {a.repeats} repeats: {min(c[0] for c in cv):.3f} to {max(c[0] for c in cv):.3f}")
    if a.save:
        Path(a.save).write_text(json.dumps({"thresholds": [float(x) for x in th_all]}, indent=2))
        print("saved", a.save)


if __name__ == "__main__":
    main()
