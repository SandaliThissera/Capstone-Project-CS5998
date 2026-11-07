import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score


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


def screening_metrics(y_true, probs):
    """Binary views of the 5-grade problem: any DR (grade >= 1) and referable DR (grade >= 2)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    pred = probs.argmax(1)
    out = {}
    for name, k in (("anydr", 1), ("referable", 2)):
        pos, pred_pos = y_true >= k, pred >= k
        tp, fn = (pos & pred_pos).sum(), (pos & ~pred_pos).sum()
        tn, fp = (~pos & ~pred_pos).sum(), (~pos & pred_pos).sum()
        out[f"{name}_sens"] = float(tp / max(tp + fn, 1))
        out[f"{name}_spec"] = float(tn / max(tn + fp, 1))
        out[f"{name}_auc"] = float(roc_auc_score(pos, probs[:, k:].sum(1)))
    return out
