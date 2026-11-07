import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--last-k", type=int, default=5)
    ap.add_argument("--out", default="reports/results_logs.csv")
    a = ap.parse_args()

    rows = []
    for d in sorted(Path(a.runs_dir).iterdir()):
        lp = d / "log.csv"
        if not lp.exists():
            continue
        log = pd.read_csv(lp)
        if log.empty:
            continue
        tail = log.tail(a.last_k)
        rec = np.array([json.loads(x) for x in tail["recall_per_class"]])
        row = {"run": d.name, "epochs": int(log["epoch"].iloc[-1]),
               "acc": tail["val_acc"].mean(), "qwk": tail["val_qwk"].mean(),
               "qwk_sd": tail["val_qwk"].std(), "macro_f1": tail["val_macro_f1"].mean(),
               "qwk_final_epoch": float(log["val_qwk"].iloc[-1])}
        for i, v in enumerate(rec.mean(axis=0)):
            row[f"recall_{i}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Mean of the last {a.last_k} epochs per run (validation split)")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
