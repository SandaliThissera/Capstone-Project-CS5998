import numpy as np
import timm
import torch
from torch import nn

from src.evaluation.metrics import decode_regression, scores_from_probs, scores_from_regression


def is_regression(cfg):
    return cfg["model"].get("head", "classification") == "regression"


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    use_pretrained = m["pretrained"] if pretrained is None else pretrained
    n_out = 1 if is_regression(cfg) else m["num_classes"]
    return timm.create_model(m["arch"], pretrained=use_pretrained, num_classes=n_out)


class WeightedSmoothL1(nn.Module):
    """SmoothL1 on the grade value, with optional per-sample class weights (weighted mean)."""

    def __init__(self, class_w=None):
        super().__init__()
        self.has_w = class_w is not None
        self.register_buffer("w", class_w if self.has_w else torch.zeros(1))

    def forward(self, out, y):
        loss = nn.functional.smooth_l1_loss(out.squeeze(1).float(), y.float(), reduction="none")
        if self.has_w:
            w = self.w[y]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def build_criterion(cfg, class_w=None):
    if is_regression(cfg):
        return WeightedSmoothL1(class_w)
    return nn.CrossEntropyLoss(weight=class_w)


def softmax_np(out):
    e = np.exp(out - out.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def decode(cfg, out):
    return decode_regression(out[:, 0]) if is_regression(cfg) else out.argmax(1)


def get_scores(cfg, out):
    if is_regression(cfg):
        return scores_from_regression(out[:, 0])
    return scores_from_probs(softmax_np(out))


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    """Returns (true labels, raw model outputs, mean loss or None)."""
    model.eval()
    ys, outs, total, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        out = out.float()
        if criterion is not None:
            total += criterion(out, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy())
        outs.append(out.cpu().numpy())
    return np.concatenate(ys), np.concatenate(outs), (total / n if criterion is not None else None)
