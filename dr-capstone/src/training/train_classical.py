import argparse, json
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

from src.data.features import extract_features
from src.evaluation.metrics import compute_metrics


def _feat(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return extract_features(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="configs/splits_aptos.csv")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--with-test", action="store_true",
                    help="also evaluate on the test split (final reporting only)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.splits)

    feat_path = out / "features.csv"
    if feat_path.exists():
        feats = pd.read_csv(feat_path)
    else:
        paths = [Path(a.cache) / f"{i}.png" for i in df["id_code"]]
        with Pool(a.workers) as pool:
            res = list(tqdm(pool.imap(_feat, paths, chunksize=16), total=len(paths)))
        feats = pd.DataFrame(res)
        feats.insert(0, "id_code", df["id_code"].values)
        feats.to_csv(feat_path, index=False)

    data = df.merge(feats, on="id_code")
    cols = [c for c in feats.columns if c != "id_code"]
    tr, va = data[data.split == "train"], data[data.split == "val"]

    clf = HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, learning_rate=0.05,
                                         early_stopping=False, random_state=a.seed)
    clf.fit(tr[cols], tr["diagnosis"])
    pred = clf.predict(va[cols])

    mt = compute_metrics(va["diagnosis"].values, pred)
    print(json.dumps(mt, indent=2))
    (out / "metrics_val.json").write_text(json.dumps(mt, indent=2))
    pd.DataFrame({"id_code": va["id_code"].values, "y_true": va["diagnosis"].values,
                  "y_pred": pred}).to_csv(out / "preds_val.csv", index=False)
    if a.with_test:
        te = data[data.split == "test"]
        pred_te = clf.predict(te[cols])
        mt_te = compute_metrics(te["diagnosis"].values, pred_te)
        print("TEST split:")
        print(json.dumps(mt_te, indent=2))
        (out / "metrics_test.json").write_text(json.dumps(mt_te, indent=2))
        pd.DataFrame({"id_code": te["id_code"].values, "y_true": te["diagnosis"].values,
                      "y_pred": pred_te}).to_csv(out / "preds_test.csv", index=False)
    print("features:", len(cols), "| saved to", out)


if __name__ == "__main__":
    main()
