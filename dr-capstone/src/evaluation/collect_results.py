import argparse, json
from pathlib import Path

import pandas as pd
import yaml

FILES = (("last", "metrics_val_last.json"), ("best", "metrics_val.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", default="reports/results_val.csv")
    a = ap.parse_args()

    rows = []
    for d in sorted(Path(a.runs_dir).iterdir()):
        cp = d / "config.yaml"
        cfg = yaml.safe_load(cp.read_text()) if cp.exists() else None
        for ckpt, fname in FILES:
            mp = d / fname
            if not mp.exists():
                continue
            row = {"run": d.name, "ckpt": ckpt if cfg else "-"}
            if cfg:
                row.update(arch=cfg["model"]["arch"], image_size=cfg["data"]["image_size"],
                           enhance=cfg["data"]["enhance"], class_weights=cfg["train"]["class_weights"],
                           batch=cfg["train"]["batch_size"], seed=cfg["seed"],
                           head=cfg["model"].get("head", "classification"),
                           cw_power=cfg["train"].get("class_weight_power", 1.0))
            mt = json.loads(mp.read_text())
            for i, r in enumerate(mt.pop("recall_per_class", [])):
                mt[f"recall_{i}"] = r
            row.update(mt)
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    show = [c for c in ("run", "ckpt", "image_size", "enhance", "class_weights", "cw_power", "head", "qwk", "acc", "macro_f1",
                        "referable_sens", "referable_spec", "referable_auc",
                        "recall_0", "recall_1", "recall_2", "recall_3", "recall_4") if c in df.columns]
    print(df[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
