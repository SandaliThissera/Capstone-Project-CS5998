import argparse, json, re
from pathlib import Path

import pandas as pd

METRICS = ["qwk", "acc", "macro_f1", "referable_sens", "referable_spec", "referable_auc",
           "recall_0", "recall_1", "recall_2", "recall_3", "recall_4"]


def base_name(run):
    return re.sub(r"_s\d+$", "", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these base config names")
    ap.add_argument("--ckpt", default="last", choices=["last", "best"])
    ap.add_argument("--out", default="reports/results_seeds.csv")
    a = ap.parse_args()

    fname = "metrics_val_last.json" if a.ckpt == "last" else "metrics_val.json"
    rows = []
    for d in sorted(Path(a.runs_dir).iterdir()):
        mp = d / fname
        if not mp.exists():
            continue
        b = base_name(d.name)
        if a.only and b not in a.only:
            continue
        mt = json.loads(mp.read_text())
        for i, r in enumerate(mt.pop("recall_per_class", [])):
            mt[f"recall_{i}"] = r
        rows.append({"config": b, "run": d.name, **mt})

    df = pd.DataFrame(rows)
    out = []
    for cfg, g in df.groupby("config"):
        row = {"config": cfg, "n_seeds": len(g)}
        for m in METRICS:
            if m in g and g[m].notna().all():
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_sd"] = g[m].std(ddof=1) if len(g) > 1 else float("nan")
                row[f"{m}_min"], row[f"{m}_max"] = g[m].min(), g[m].max()
        out.append(row)
    res = pd.DataFrame(out).sort_values("qwk_mean", ascending=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    print(f"Validation, {a.ckpt}-epoch checkpoint; mean (sd) over seeds. Runs: "
          + ", ".join(f"{r.config}={r.n_seeds}" for r in res.itertuples()))
    show = pd.DataFrame({"config": res["config"], "n": res["n_seeds"]})
    for m in ("qwk", "acc", "macro_f1", "referable_sens", "referable_spec", "referable_auc",
              "recall_3", "recall_4"):
        if f"{m}_mean" in res:
            show[m] = [f"{x:.3f} ({s:.3f})" if s == s else f"{x:.3f} (n=1)"
                       for x, s in zip(res[f"{m}_mean"], res[f"{m}_sd"])]
    print(show.to_string(index=False))
    print("\nWith 3 seeds the sd itself is rough; also check min/max in", a.out)


if __name__ == "__main__":
    main()
