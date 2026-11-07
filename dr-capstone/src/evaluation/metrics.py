import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


def compute_metrics(y_true, y_pred, num_classes=5):
    labels = list(range(num_classes))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "acc": float((y_true == y_pred).mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=labels,
                                         zero_division=0).round(3).tolist(),
    }
