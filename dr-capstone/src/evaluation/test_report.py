import argparse, json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from src.evaluation.ensemble_eval import load_probs, operating_points

LABELS = list(range(5))
NAMES = {"qwk": "QWK", "acc": "Accuracy", "macro_f1": "Macro-F1", "referable_auc": "Referable AUC",
         "ref_sens_at_cutoff": "Referable sensitivity (fixed cut-off)",
         "ref_spec_at_cutoff": "Referable specificity (fixed cut-off)",
         **{f"recall_{k}": f"Recall, grade {k}" for k in LABELS}}


def metrics_at(y, probs, cutoff):
    pred = probs.argmax(1)
    pos = y >= 2
    p_ref = probs[:, 2:].sum(1)
    flag = p_ref >= cutoff
    m = {"qwk": cohen_kappa_score(y, pred, weights="quadratic", labels=LABELS),
         "acc": float((pred == y).mean()),
         "macro_f1": f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)}
    for k in LABELS:
        m[f"recall_{k}"] = float((pred[y == k] == k).mean()) if (y == k).any() else float("nan")
    m["referable_auc"] = roc_auc_score(pos, p_ref) if pos.any() and (~pos).any() else float("nan")
    m["ref_sens_at_cutoff"] = float((flag & pos).sum() / max(pos.sum(), 1))
    m["ref_spec_at_cutoff"] = float((~flag & ~pos).sum() / max((~pos).sum(), 1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders averaged into one model")
    ap.add_argument("--val-file", default="probs_val_last.csv")
    ap.add_argument("--test-file", default="probs_test_last.csv")
    ap.add_argument("--target-sens", type=float, default=0.95,
                    help="referable-DR sensitivity the cut-off is chosen for, on validation")
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    for r in a.runs:
        for f in (a.val_file, a.test_file):
            if not (Path(a.runs_dir) / r / f).exists():
                raise SystemExit(f"missing {r}/{f}. Run src.evaluation.evaluate with --split val|test --ckpt last first.")

    yv, pv = load_probs(a.runs_dir, a.runs, a.val_file)
    (_, op), = operating_points(yv, pv, [a.target_sens])
    if op is None:
        raise SystemExit(f"target sensitivity {a.target_sens} is not reachable on validation")
    cutoff = op[0]

    yt, pt = load_probs(a.runs_dir, a.runs, a.test_file)
    n = len(yt)
    est = metrics_at(yt, pt, cutoff)
    rng = np.random.default_rng(a.seed)
    boots = {k: [] for k in est}
    for _ in range(a.boots):
        idx = rng.integers(0, n, n)
        for k, v in metrics_at(yt[idx], pt[idx], cutoff).items():
            boots[k].append(v)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in boots.items()}

    pos = yt >= 2
    flag = pt[:, 2:].sum(1) >= cutoff
    caught, missed = int((flag & pos).sum()), int((~flag & pos).sum())
    false_alarms, n_neg = int((flag & ~pos).sum()), int((~pos).sum())

    print(f"Final evaluation on the TEST split | model: average of {len(a.runs)} run(s): {', '.join(a.runs)} | {n} images")
    print(f"Referable-DR cut-off chosen on validation for sensitivity >= {a.target_sens}: {cutoff:.4f} "
          f"(validation: sens {op[1]:.3f}, spec {op[2]:.3f})\n")
    print(f"{'metric':40s} {'estimate':>8s}   95% CI")
    for k in est:
        print(f"{NAMES[k]:40s} {est[k]:8.3f}   {ci[k][0]:.3f} to {ci[k][1]:.3f}")
    print(f"\nAt the fixed cut-off on test: caught {caught} of {int(pos.sum())} referable cases (missed {missed}); "
          f"{false_alarms} false alarms among {n_neg} non-referable images.")
    print("CIs: percentile bootstrap over test images ({} resamples). They do not include seed-to-seed variation "
          "or the uncertainty of the validation-chosen cut-off.".format(a.boots))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "runs": a.runs, "n_test": n, "target_sens": a.target_sens, "cutoff_from_val": cutoff,
            "val_at_cutoff": {"sens": op[1], "spec": op[2]},
            "test_counts": {"referable": int(pos.sum()), "caught": caught, "missed": missed,
                            "non_referable": n_neg, "false_alarms": false_alarms},
            "test": {k: {"estimate": est[k], "ci_low": ci[k][0], "ci_high": ci[k][1]} for k in est}}, indent=2))
        print("saved", a.out)


if __name__ == "__main__":
    main()
