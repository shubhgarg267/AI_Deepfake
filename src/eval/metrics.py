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


def compute_all(labels, probs):
    return {
        "accuracy": accuracy(labels, probs),
        "f1": f1(labels, probs),
        "auc": auc(labels, probs),
    }
