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
    """Local contrast enhancement (subtract a blurred copy), border set to neutral grey."""
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (0, 0), max(w * sigma_frac, 1))
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.48), 255, -1)
    out[mask == 0] = 128
    return out
