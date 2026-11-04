import numpy as np


def compute_aurac(ground_truth_list, uncertainty_list):
    """
    Computes AURAC: Area Under the Risk-Accuracy Curve.

    Args:
        ground_truth_list: List of 0 (correct) and 1 (incorrect) labels.
        uncertainty_list: List of uncertainty values (0.0 - 1.0).

    Returns:
        AURAC score.
    """
    sorted_indices = np.argsort(uncertainty_list)
    sorted_truths = np.array(ground_truth_list)[sorted_indices]
    risk = np.cumsum(sorted_truths) / np.arange(1, len(sorted_truths) + 1)
    accuracy = 1 - risk
    aurac = np.trapz(accuracy, dx=1.0 / len(accuracy))
    return round(aurac * 100, 1)
