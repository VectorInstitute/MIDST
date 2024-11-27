import argparse
import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

FPR_THRESHOLD_LIST = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]


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


def write_file(file:str, content: str):
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)

def image_to_html(fig):
    """Converts a matplotlib plot to SVG"""
    iostring = io.StringIO()
    fig.savefig(iostring, format="svg", bbox_inches=0, dpi=300, pad_inches=0.2)
    iostring.seek(0)

    return iostring.read()

def generate_table(scores):
    table = pd.DataFrame(scores)
    replace_column = {
        "accuracy":  "Accuracy",
        "AUC": "AUC-ROC",
        "MIA": "MIA",
        "TPR_FPR_10": "TPR @ 0.001 FPR",
        "TPR_FPR_100": "TPR @ 0.01 FPR",
        "TPR_FPR_500": "TPR @ 0.05 FPR",
        "TPR_FPR_1000": "TPR @ 0.1 FPR",
        "TPR_FPR_1500": "TPR @ 0.15 FPR",
        "TPR_FPR_2000": "TPR @ 0.2 FPR",
    }
    table.columns = [replace_column[c] for c in table.columns]

    return table


def generate_roc(fpr: np.ndarray, tpr: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,3.5))

    ax2.semilogx()
    ax2.semilogy()
    ax2.set_xlim(1e-5,1)
    ax2.set_ylim(1e-5,1)
    ax2.set_xlabel("False Positive Rate")
    ax2.plot([0, 1], [0, 1], ls=':', color='grey')

    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.plot([0,1], [0,1], ls=':', color='grey')

    ax1.plot(fpr, tpr)
    ax2.plot(fpr, tpr)

    # Adjust layout to ensure no labels are cut off
    plt.subplots_adjust(bottom=0.2)  # Add extra space at the bottom

    return fig


def score(solutions: np.ndarray, predictions: np.ndarray) -> float:
    return get_tpr_at_fpr(solutions, predictions)


def generate_detailed_results(html_file: str, predictions: np.ndarray, solutions: np.ndarray) -> None:
    scores = {}
    for max_fpr in FPR_THRESHOLD_LIST:
        scores[f"TPR_FPR_{int(1e4 * max_fpr)}"] = [get_tpr_at_fpr(solutions, predictions, max_fpr=max_fpr)]
    fpr, tpr, _ = roc_curve(solutions, predictions)
    scores["AUC"] = [roc_auc_score(solutions, predictions)]
    scores["MIA"] = [np.max(tpr - fpr)]
    # This is the balanced accuracy, which coincides with accuracy for balanced classes
    scores["accuracy"] = [np.max(1 - (fpr + (1 - tpr)) / 2)]

    table = generate_table(scores)

    roc_figure = generate_roc(fpr, tpr)

    # Generate the HTML document.
    css = '''
    body {
        background-color: #ffffff;
    }
    h1 {
        text-align: center;
    }
    h2 {
        text-align: center;
    }
    div {
        white-space: normal;
        text-align: center;
    }
    table {
      border-collapse: collapse;
      margin: auto;
    }
    table > :is(thead, tbody) > tr > :is(th, td) {
      padding: 5px;
    }
    table > thead > tr > :is(th, td) {
      border-top:    2px solid; /* \toprule */
      border-bottom: 1px solid; /* \midrule */
    }
    table > tbody > tr:last-child > :is(th, td) {
      border-bottom: 2px solid; /* \bottomrule */
    }'''

    html = f'''<!DOCTYPE html>
    <html>
    <head>
        <title>MIDST - Detailed scores</title>
        <style>
        {css}
        </style>
    </head>
    <body>
    <h1>MIDST - Detailed Results</h1>

    <h2>Metric Scores</h2>
    <div>
    {table.to_html(border=0, float_format='{:0.4f}'.format, escape=False, index=False)}
    </div>

    <h2>ROC Curve</h2>
    <div>
    {image_to_html(roc_figure)}
    </div>

    </body></html>
    '''

    write_file(html_file, html)


def get_scores(dev_or_final: str):
    base_solutions_dir = os.path.join('/app/input/', 'ref')
    base_predictions_dir = os.path.join('/app/input/', 'res')
    output_dir = '/app/output/'
    score_file = os.path.join(output_dir, 'scores.json')
    html_file = os.path.join(output_dir, 'scores.html')

    solutions_dir = os.path.join(base_solutions_dir, "clavaddpm_black_box", dev_or_final)
    assert os.path.exists(solutions_dir), f"Directory {solutions_dir} does not exist. Please contact competition oragnizers."

    predictions_dir = os.path.join(base_predictions_dir, "clavaddpm_black_box", dev_or_final)
    assert os.path.exists(predictions_dir), f"Directory {predictions_dir} does not exist. \
        Ensure root of extracted submission contains clavaddpm_black_box/{dev_or_final} folder. \
        Ex: clavaddpm_black_box/{dev_or_final}/clavaddpm_#/prediction.csv"

    mapping_file = os.path.join(base_solutions_dir, "clavaddpm_mapping_final.json")
    assert os.path.exists(mapping_file), f"File {mapping_file} does not exist. Please contact competition oragnizers."

    with open(mapping_file) as f:
        mapping_data = json.load(f)

    # We compute the scores globally, across the models of the same model type. 
    # This is somewhat equivalent to having one attack (threshold) for all the attacks.
    predictions = []
    solutions  = []
    for model_id in mapping_data[f"{dev_or_final}_black_box"]:
        label_path = os.path.join(solutions_dir, model_id, "challenge_label.csv")
        assert os.path.exists(label_path), f"File {label_path} does not exist. Please contact competition oragnizers."

        pred_path = os.path.join(predictions_dir, model_id, "prediction.csv")
        assert os.path.exists(pred_path), f"File {pred_path} does not exist.\
            Ensure a predictions.csv file exists for model folders.\
            Ex: clavaddpm_black_box/{dev_or_final}/clavaddpm_#/prediction.csv"

        solutions.append(np.loadtxt(label_path, skiprows=1))
        predictions.append(np.loadtxt(pred_path))

    solutions = np.concatenate(solutions)
    predictions = np.concatenate(predictions)

    # Verify that the predictions are valid.
    assert len(predictions) == len(solutions)
    assert np.all(predictions >= 0), "Some predictions are < 0"
    assert np.all(predictions <= 1), "Some predictions are > 1"

    tpr_at_fpr = score(solutions, predictions)

    print(f"TPR at FPR at FPR == 10%", tpr_at_fpr)

    generate_detailed_results(html_file, predictions, solutions)

    with open(os.path.join(output_dir, 'scores.json'), 'w') as score_file:
        score_file.write(json.dumps({"tpr_at_fpr": tpr_at_fpr}))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dev_or_final", type=str)
    args = argparser.parse_args()

    get_scores(args.dev_or_final)
