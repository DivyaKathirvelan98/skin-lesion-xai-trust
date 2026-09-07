"""Classification and calibration metrics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                                 class_names: list) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_auroc": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
    }


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 15) -> float:
    """Standard ECE: weighted average gap between confidence and accuracy across bins."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
