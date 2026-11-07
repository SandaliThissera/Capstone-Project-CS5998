import cv2
import numpy as np


def crop_black_border(img, tol=10, min_frac=0.01):
    """Crop to the bounding box of the retina. Works on an RGB uint8 image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.where(mask.sum(axis=1) > min_frac * mask.shape[1])[0]
    cols = np.where(mask.sum(axis=0) > min_frac * mask.shape[0])[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def preprocess(img, size=512):
    img = pad_to_square(crop_black_border(img))
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def ben_graham(img, sigma_frac=1 / 30):
    """Local contrast enhancement: 4*(img - blur) + 128, computed over the retina only.

    Dark pixels count as retina only if they are enclosed by it. Dark areas that
    touch the image border (padding, corners) are background and end up neutral grey,
    and the blur never averages them in.
    """
    w = img.shape[1]
    sigma = max(w * sigma_frac, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    inv = (gray <= 10).astype(np.uint8)
    _, lab = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = inv.astype(bool) & ~np.isin(lab, border)
    valid = ((gray > 10) | holes).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))

    vf = valid.astype(np.float32)
    imgf = img.astype(np.float32)
    num = cv2.GaussianBlur(imgf * vf[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(vf, (0, 0), sigma)[..., None]
    blur = num / np.maximum(den, 1e-3)
    out = np.clip(4 * (imgf - blur) + 128, 0, 255).astype(np.uint8)
    out[valid == 0] = 128
    return out
