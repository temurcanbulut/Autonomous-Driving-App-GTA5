"""Pixel confusion-matrix metrics; absent classes do not become perfect scores."""

import numpy as np


def summarize(matrix, classes):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or (matrix < 0).any() or not matrix.sum():
        raise ValueError("A nonnegative square confusion matrix with labelled pixels is required")
    true_positive = matrix.diagonal().astype(float)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    union = support + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.full(len(classes), np.nan), where=union > 0)
    recall = np.divide(true_positive, support, out=np.full(len(classes), np.nan), where=support > 0)
    return {
        "mean_iou": float(np.nanmean(iou)),
        "pixel_accuracy": float(true_positive.sum() / matrix.sum()),
        "classes": [
            {
                "name": name,
                "iou": None if not np.isfinite(iou[i]) else float(iou[i]),
                "recall": None if not np.isfinite(recall[i]) else float(recall[i]),
                "support_pixels": int(support[i]),
            }
            for i, name in enumerate(classes)
        ],
        "confusion_matrix": matrix.tolist(),
    }
