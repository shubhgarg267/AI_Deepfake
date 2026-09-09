"""Hand-rolled accuracy/F1/AUC (avoiding a new sklearn pin, consistent with
how DCT/PSNR/SSIM were self-implemented elsewhere in this project)."""
import numpy as np


def accuracy(labels, probs, threshold=0.5):
    preds = (np.asarray(probs) > threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    return float((preds == labels).mean())


def f1(labels, probs, threshold=0.5):
    preds = (np.asarray(probs) > threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def auc(labels, probs):
    """Rank-sum (Mann-Whitney U) formula -- exact, no thresholding needed."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(probs)
    ranks = np.empty(len(probs), dtype=np.float64)
    sorted_probs = probs[order]
    i = 0
    rank = 1
    while i < len(sorted_probs):
        j = i
        while j < len(sorted_probs) and sorted_probs[j] == sorted_probs[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j

    sum_ranks_pos = ranks[labels == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def confusion_matrix(labels, probs, threshold=0.5):
    preds = (np.asarray(probs) > threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def roc_points(labels, probs):
    """Exact ROC curve: (FPR, TPR) at every unique threshold, from (0,0) to (1,1)."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    order = np.argsort(-probs)  # descending score
    labels_sorted = labels[order]
    probs_sorted = probs[order]

    points = [(0.0, 0.0)]
    tp = fp = 0
    i, n = 0, len(labels_sorted)
    while i < n:
        j = i
        while j < n and probs_sorted[j] == probs_sorted[i]:
            if labels_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        points.append((fpr, tpr))
        i = j
    return points


def eer(labels, probs):
    """Equal Error Rate via linear interpolation on the ROC curve (where FPR == FNR)."""
    points = roc_points(labels, probs)
    for k in range(len(points) - 1):
        fpr1, tpr1 = points[k]
        fpr2, tpr2 = points[k + 1]
        s1 = fpr1 - (1 - tpr1)  # FPR - FNR
        s2 = fpr2 - (1 - tpr2)
        if s1 == 0:
            return fpr1
        if s1 < 0 <= s2:
            t = -s1 / (s2 - s1) if (s2 - s1) != 0 else 0.0
            fpr_i = fpr1 + t * (fpr2 - fpr1)
            fnr_i = (1 - tpr1) + t * ((1 - tpr2) - (1 - tpr1))
            return (fpr_i + fnr_i) / 2
    fpr_last, tpr_last = points[-1]
    return (fpr_last + (1 - tpr_last)) / 2


def compute_all(labels, probs):
    return {
        "accuracy": accuracy(labels, probs),
        "f1": f1(labels, probs),
        "auc": auc(labels, probs),
    }
