from sklearn.metrics import roc_curve

def get_tpr_at_fpr(true_membership: list, predictions: list, max_fpr=0.1) -> float:
    """Calculates the best True Positive Rate when the False Positive Rate is
    at most `max_fpr`.

    Args:
        true_membership (List): A list of values in {0,1} indicating the membership of a
            challenge point. 0: "non-member", 1: "member".
        predictions (List): A list of values in the range [0,1] indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more
            confident the predictor is about the hypothesis that the challenge point is
            a member.
        max_fpr (float, optional): Threshold on the FPR. Defaults to 0.1.

    Returns:
        float: The TPR @ `max_fpr` FPR.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)

    return max(tpr[fpr < max_fpr])
