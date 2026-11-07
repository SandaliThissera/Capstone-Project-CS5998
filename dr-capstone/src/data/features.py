import cv2
import numpy as np


def retina_mask(img, tol=10):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = (gray > tol).astype(np.uint8)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))   # drop the rim
    return m.astype(bool)


def extract_features(img):
    """Handcrafted features from a preprocessed 512 px RGB uint8 image. Returns a dict."""
    mask = retina_mask(img)
    if mask.sum() < 1000:
        mask = np.ones(mask.shape, bool)
    f = {}

    for i, ch in enumerate("rgb"):
        v = img[..., i][mask].astype(np.float32)
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        f.update({f"{ch}_mean": v.mean(), f"{ch}_std": v.std(),
                  f"{ch}_p5": p5, f"{ch}_p50": p50, f"{ch}_p95": p95})

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f["dark_frac"] = (gray[mask] < 30).mean()
    f["lap_var"] = cv2.Laplacian(gray, cv2.CV_32F)[mask].var()

    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img[..., 1])
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ell(15))[mask]      # thin dark structures: vessels
    bh_big = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ell(31))[mask]  # larger dark spots
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ell(25))[mask]        # small bright spots

    f["vessel_bh_mean"] = bh.mean()
    for t in (8, 12, 20):
        f[f"vessel_frac_{t}"] = (bh > t).mean()
    f["dark_big_mean"] = bh_big.mean()
    for t in (15, 25, 40):
        f[f"dark_big_frac_{t}"] = (bh_big > t).mean()
    f["bright_th_mean"] = th.mean()
    for t in (20, 30, 45):
        f[f"bright_frac_{t}"] = (th > t).mean()
    return {k: float(v) for k, v in f.items()}
