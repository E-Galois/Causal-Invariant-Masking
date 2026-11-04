from sklearn.metrics import roc_auc_score


def compute_auroc(ground_truth_list, uncertainty_list):
    """
    Computes AUROC: Measures how well uncertainty values distinguish between correct and incorrect predictions.

    Args:
        ground_truth_list: List of 0 (correct) and 1 (incorrect) labels.
        uncertainty_list: List of uncertainty values (0.0 - 1.0).

    Returns:
        AUROC score.
    """
    auroc = roc_auc_score(ground_truth_list, uncertainty_list)
    return round(auroc * 100.0, 1)
