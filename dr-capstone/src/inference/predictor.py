"""Shared inference code: preprocessing, 3-model ensemble, Grad-CAM overlay."""
import base64
from pathlib import Path

import cv2
import numpy as np

from src.data.preprocess import preprocess

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LOW_CONF = 0.6  # display heuristic only; not tuned or validated


def prepare(rgb, size=512, already_preprocessed=False):
    """RGB uint8 -> (preprocessed RGB image, normalised CHW float32). Same steps as the training cache."""
    img = rgb if already_preprocessed else preprocess(rgb, size)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return img, np.ascontiguousarray(x.transpose(2, 0, 1))


def overlay(img, cam, alpha=0.45):
    cam = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)


def png_b64(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


class Predictor:
    def __init__(self, weights_path, device="cpu"):
        import timm, torch
        self.torch, self.device = torch, torch.device(device)
        ck = torch.load(weights_path, map_location=self.device)
        self.cutoff, self.size, self.runs = ck["cutoff"], ck["size"], ck["runs"]
        self.models = []
        for sd in ck["state_dicts"]:
            m = timm.create_model(ck["arch"], pretrained=False, num_classes=5)
            m.load_state_dict(sd)
            self.models.append(m.to(self.device).eval())

    def _cam(self, model, x, cls):
        torch = self.torch
        with torch.enable_grad():
            feats = model.forward_features(x)
            feats.retain_grad()
            out = model.forward_head(feats)
            model.zero_grad()
            out[0, cls].backward()
            w = feats.grad.mean((2, 3), keepdim=True)
            cam = torch.relu((w * feats).sum(1))[0].detach().cpu().numpy()
        return cam / (cam.max() + 1e-8)

    def predict(self, rgb, with_cam=True, already_preprocessed=False):
        torch = self.torch
        img, x = prepare(rgb, self.size, already_preprocessed)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = np.mean([torch.softmax(m(xt), 1)[0].cpu().numpy() for m in self.models], axis=0)
        grade = int(probs.argmax())
        p_ref = float(probs[2:].sum())
        res = {"grade": grade, "grade_name": GRADES[grade], "probs": [float(p) for p in probs],
               "confidence": float(probs.max()), "low_confidence": bool(probs.max() < LOW_CONF),
               "p_referable": p_ref, "cutoff": float(self.cutoff), "referable": bool(p_ref >= self.cutoff),
               "n_models": len(self.models), "image_png_b64": png_b64(img)}
        if with_cam:
            cams = [cv2.resize(self._cam(m, xt.clone(), grade), (self.size, self.size)) for m in self.models]
            res["cam_png_b64"] = png_b64(overlay(img, np.mean(cams, axis=0)))
        return res
