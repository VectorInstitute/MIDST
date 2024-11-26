import argparse
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

def get_scores(dev_or_final: str):
    base_solutions_dir = os.path.join('/app/input/', 'ref')
    base_predictions_dir = os.path.join('/app/input/', 'res')
    output_dir = '/app/output/'


    tpr_at_fpr_list = []

    for model_type in ["tabddpm_black_box", "tabsyn_black_box"]:
        # Skip iteration if model_type directory not in base predictions directory
        if model_type not in os.listdir(base_predictions_dir): continue

        model_name = model_type.split("_")[0]

        solutions_dir = os.path.join(base_solutions_dir, model_type, dev_or_final)
        assert os.path.exists(solutions_dir), f"Directory {solutions_dir} does not exist. Please contact competition oragnizers."

        predictions_dir = os.path.join(base_predictions_dir, model_type, dev_or_final)
        assert os.path.exists(predictions_dir), f"Directory {predictions_dir} does not exist. \
            Ensure root of extracted submission contains {model_type}/{dev_or_final} folder. \
            Ex: {model_type}/{dev_or_final}/{model_name}_#/predictions.csv"

        mapping_file = os.path.join(base_solutions_dir, f"{model_name}_mapping_final.json")
        assert os.path.exists(mapping_file), f"File {mapping_file} does not exist. Please contact competition oragnizers."
        
        with open(mapping_file) as f:
            mapping_data = json.load(f)

        # We compute the scores globally, across the models of the same model type. 
        # This is somewhat equivalent to having one attack (threshold) for all the attacks.
        # Load the predictions.
        predictions = []
        solutions  = []
        for model_id in mapping_data[f"{dev_or_final}_black_box"]:
            label_path = os.path.join(solutions_dir, model_id, "challenge_label.csv")
            assert os.path.exists(label_path), f"File {label_path} does not exist. Please contact competition oragnizers."

            pred_path = os.path.join(predictions_dir, model_id, "prediction.csv")
            assert os.path.exists(pred_path), f"File {pred_path} does not exist.\
                Ensure a predictions.csv file exists for model folders.\
                Ex: {model_type}/{dev_or_final}/{model_name}_#/predictions.csv"

            solutions.append(np.loadtxt(label_path, skiprows=1))
            predictions.append(np.loadtxt(pred_path))

        solutions = np.concatenate(solutions)
        predictions = np.concatenate(predictions)

        # Verify that the predictions are valid.
        assert len(predictions) == len(solutions)
        assert np.all(predictions >= 0), "Some predictions are < 0"
        assert np.all(predictions <= 1), "Some predictions are > 1"

        tpr_at_fpr = score(solutions, predictions)
        tpr_at_fpr_list.append(tpr_at_fpr)

        print(f"{model_type.split('_')[0]} TPR at FPR at FPR == 10%", tpr_at_fpr)

    assert len(tpr_at_fpr_list) > 0, "We expect to have at least one model type present of TabDDPM and TabSyn."

    with open(os.path.join(output_dir, 'scores.json'), 'w') as score_file:
        score_file.write(json.dumps({"tpr_at_fpr": max(tpr_at_fpr_list)}))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dev_or_final", type=str)
    args = argparser.parse_args()

    get_scores(args.dev_or_final)
