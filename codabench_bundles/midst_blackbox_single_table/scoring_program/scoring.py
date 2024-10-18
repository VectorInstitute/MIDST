import os
import json
import numpy as np

from sklearn.metrics import roc_curve


def get_tpr_at_fpr(true_membership: np.ndarray, predictions: np.ndarray, max_fpr=0.1) -> float:
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


def score(solutions: np.ndarray, predictions: np.ndarray) -> float:
    return get_tpr_at_fpr(solutions, predictions) 


if __name__ == "__main__":
    solutions_dir = os.path.join('/app/input/', 'ref')
    predictions_dir = os.path.join('/app/input/', 'res')
    output_dir = '/app/output/'

    assert len(os.listdir(predictions_dir)) == len(os.listdir(predictions_dir))

    # We compute the scores globally, across the models. This is somewhat equivalent to having
    # one attack (threshold) for all the attacks.
    # Load the predictions.
    predictions = []
    solutions  = []
    for model_id in os.listdir(solutions_dir):
        solutions.append(np.loadtxt(os.path.join(solutions_dir, model_id, "challenge_label.csv"), skiprows=1))
        predictions.append(np.loadtxt(os.path.join(predictions_dir, model_id, "prediction.csv")))

    solutions = np.concatenate(solutions)
    predictions = np.concatenate(predictions)

    # Verify that the predictions are valid.
    assert len(predictions) == len(solutions)
    assert np.all(predictions >= 0), "Some predictions are < 0"
    assert np.all(predictions <= 1), "Some predictions are > 1"

    tpr_at_fpr = score(solutions, predictions)
    print("TPR at FPR", tpr_at_fpr)

    with open(os.path.join(output_dir, 'scores.json'), 'w') as score_file:
        score_file.write(json.dumps({"tpr_at_fpr": tpr_at_fpr}))
