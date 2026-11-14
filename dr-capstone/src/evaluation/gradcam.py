import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = [(3, 2, "Grade 3 predicted as 2"), (4, 2, "Grade 4 predicted as 2"),
          (2, 1, "Grade 2 predicted as 1"), (1, 2, "Grade 1 predicted as 2")]
CONTROLS = [2, 3, 4]
PCOLS = [f"p{i}" for i in range(5)]


def load_ensemble(runs_dir, runs, fname):
    dfs = [pd.read_csv(Path(runs_dir) / r / fname) for r in runs]
    for d in dfs[1:]:
        assert (d["id_code"].values == dfs[0]["id_code"].values).all(), "runs list images in a different order"
    p = np.mean([d[PCOLS].values for d in dfs], axis=0)
    out = dfs[0][["id_code", "y_true"]].copy()
    out[PCOLS] = p
    out["y_pred"] = p.argmax(1)
    out["conf"] = p.max(1)
    out["p_ref"] = p[:, 2:].sum(1)
    return out


def cutoff_summary(df, cutoff):
    flag = df["p_ref"] >= cutoff
    pos = df["y_true"] >= 2
    missed = df[pos & ~flag]["y_true"].value_counts().reindex([2, 3, 4], fill_value=0)
    fa = df[~pos & flag]["y_true"].value_counts().reindex([0, 1], fill_value=0)
    print(f"At cut-off {cutoff:.4f}: missed referable by true grade -> " +
          ", ".join(f"G{g}: {n}" for g, n in missed.items()) + f" (total {int(missed.sum())})")
    print("False alarms by true grade -> " + ", ".join(f"G{g}: {n}" for g, n in fa.items()) +
          f" (total {int(fa.sum())})")


def load_models(runs_dir, runs, device):
    import torch, yaml
    from src.training.engine import build_model
    models, cfg0 = [], None
    for r in runs:
        cfg = yaml.safe_load(open(Path(runs_dir) / r / "config.yaml"))
        cfg0 = cfg0 or cfg
        m = build_model(cfg, pretrained=False).to(device)
        m.load_state_dict(torch.load(Path(runs_dir) / r / "last.pt", map_location=device)["model"])
        models.append(m.eval())
    return models, cfg0


def cam_for(model, x, cls):
    import torch
    feats = model.forward_features(x)
    feats.retain_grad()
    out = model.forward_head(feats)
    model.zero_grad()
    out[0, cls].backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1))[0].detach().cpu().numpy()
    return cam / (cam.max() + 1e-8)


def draw_group(title, rows, models, cfg, device, fig_path):
    import cv2, torch
    import matplotlib.pyplot as plt
    from src.data.dataset import build_transforms
    d = cfg["data"]
    tf = build_transforms(False, d["image_size"])
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for j, (_, r) in enumerate(rows.iterrows()):
        img = cv2.cvtColor(cv2.imread(str(Path(d["cache_dir"]) / f"{r.id_code}.png")), cv2.COLOR_BGR2RGB)
        x = tf(image=img)["image"].unsqueeze(0).to(device)
        cams = [cam_for(m, x.clone(), int(r.y_pred)) for m in models]
        cam = np.mean([cv2.resize(c, (img.shape[1], img.shape[0])) for c in cams], axis=0)
        axes[0, j].imshow(img)
        axes[1, j].imshow(img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        axes[0, j].set_title(f"{r.id_code[:7]}\ntrue {int(r.y_true)} / pred {int(r.y_pred)}\np={r.conf:.2f}", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print("saved", fig_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cutoff-json", default="reports/test_final_ensemble.json")
    ap.add_argument("--n", type=int, default=6, help="images per figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    df = load_ensemble(a.runs_dir, a.runs, "probs_test_last.csv")
    err = df[df.y_true != df.y_pred]
    Path("reports").mkdir(exist_ok=True)
    err.to_csv("reports/test_errors.csv", index=False)
    print(f"{len(err)} errors of {len(df)}; saved reports/test_errors.csv")
    adj = int((err.y_true - err.y_pred).abs().eq(1).sum())
    print(f"adjacent-grade errors: {adj} of {len(err)}; predicted lower than true: {int((err.y_pred < err.y_true).sum())}")
    cutoff_summary(df, json.load(open(a.cutoff_json))["cutoff_from_val"])
    if a.no_figures:
        return

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cfg = load_models(a.runs_dir, a.runs, device)
    rng = np.random.default_rng(a.seed)
    for t, p, title in GROUPS:
        sub = df[(df.y_true == t) & (df.y_pred == p)]
        rows = sub.iloc[rng.permutation(len(sub))[:a.n]]
        draw_group(f"{title} ({len(sub)} images; random sample of {len(rows)})", rows, models, cfg, device,
                   f"{a.out_dir}/gradcam_err_{t}to{p}.png")
    for g in CONTROLS:
        sub = df[(df.y_true == g) & (df.y_pred == g)]
        rows = sub.iloc[rng.permutation(len(sub))[:a.n]]
        draw_group(f"Grade {g} correctly classified (random sample of {len(rows)})", rows, models, cfg, device,
                   f"{a.out_dir}/gradcam_ok_{g}.png")


if __name__ == "__main__":
    main()
