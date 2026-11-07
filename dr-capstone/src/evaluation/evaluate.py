import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset, build_transforms
from src.evaluation.metrics import compute_metrics, screening_metrics


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        ys.append(y.numpy())
        ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--fig-out", default=None)
    a = ap.parse_args()
    if a.split == "test":
        print("NOTE: test split - use this only for final reporting, not for tuning.")

    run_dir = Path(a.run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    d, m = cfg["data"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(m["arch"], pretrained=False, num_classes=m["num_classes"]).to(device)
    if a.ckpt == "last":
        state = torch.load(run_dir / "last.pt", map_location=device)["model"]
    else:
        state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state)
    tag = "" if a.ckpt == "best" else "_last"

    ds = DRDataset(d["splits"], a.split, d["cache_dir"], build_transforms(False, d["image_size"]), d["enhance"])
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    y, probs = predict_probs(model, dl, device)
    pred = probs.argmax(1)

    labels = list(range(m["num_classes"]))
    mt = compute_metrics(y, pred, m["num_classes"])
    mt.update(screening_metrics(y, probs))
    print(json.dumps(mt, indent=2))
    (run_dir / f"metrics_{a.split}{tag}.json").write_text(json.dumps(mt, indent=2))

    out_df = pd.DataFrame(probs, columns=[f"p{i}" for i in labels])
    out_df.insert(0, "id_code", ds.df["id_code"].values)
    out_df.insert(1, "y_true", y)
    out_df.to_csv(run_dir / f"probs_{a.split}{tag}.csv", index=False)

    cm = confusion_matrix(y, pred, labels=labels)
    frac = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(labels); ax.set_yticks(labels)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    ax.set_title(f"{cfg['name']} - {a.split}{tag} (QWK {mt['qwk']:.3f})")
    plt.tight_layout()
    out = Path(a.fig_out) if a.fig_out else run_dir / f"confusion_{a.split}{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
