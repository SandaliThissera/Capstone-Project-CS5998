import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score

THRESHOLDS = (0.5, 1.5, 2.5, 3.5)


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


def decode_regression(values, thresholds=THRESHOLDS):
    """Turn a continuous grade estimate into a class: <0.5 -> 0, 0.5-1.5 -> 1, ..., >=3.5 -> 4."""
    return np.digitize(np.asarray(values), thresholds)


def scores_from_probs(probs):
    probs = np.asarray(probs)
    return {1: probs[:, 1:].sum(1), 2: probs[:, 2:].sum(1)}


def scores_from_regression(values):
    values = np.asarray(values)
    return {1: values, 2: values}


def screening_metrics(y_true, pred, scores):
    """Binary views of the 5-grade problem: any DR (grade >= 1) and referable DR (grade >= 2).

    scores maps 1 and 2 to a score where higher means more likely positive (used for AUC).
    """
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    out = {}
    for name, k in (("anydr", 1), ("referable", 2)):
        pos, pred_pos = y_true >= k, pred >= k
        tp, fn = (pos & pred_pos).sum(), (pos & ~pred_pos).sum()
        tn, fp = (~pos & ~pred_pos).sum(), (~pos & pred_pos).sum()
        out[f"{name}_sens"] = float(tp / max(tp + fn, 1))
        out[f"{name}_spec"] = float(tn / max(tn + fp, 1))
        out[f"{name}_auc"] = float(roc_auc_score(pos, scores[k]))
    return out
