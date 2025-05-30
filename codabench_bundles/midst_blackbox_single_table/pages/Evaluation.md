## Evaluation

Submissions will be ranked based on their performance in black-box membership inference against the provided models.

There are three challenge sets: train, dev, and final. For models in train, we reveal the full training dataset, and consequently the ground truth membership data for challenge points. These models can be used by participants to develop their attacks. For models in the dev and final sets, no ground truth is revealed and participants must submit their membership predictions for challenge points.

During the competition, there will be a live scoreboard based on the dev challenges. The final ranking will be decided on the final set; scoring for this dataset will be withheld until the competition ends.

For each challenge point, the submission must provide a value, indicating the confidence level with which the challenge point is a member. Each value must be a floating point number in the range [0.0, 1.0], where 1.0 indicates certainty that the challenge point is a member, and 0.0 indicates certainty that it is a non-member.

Submissions will be evaluated according to their True Positive Rate at 10% False Positive Rate (i.e. TPR @ 0.1 FPR). In this context, positive challenge points are members and negative challenge points are non-members. For each submission, the scoring program concatenates the confidence values for all models (dev and final treated separately) and compares these to the reference ground truth. The scoring program determines the minimum confidence threshold for membership such that at most 10% of the non-member challenge points are incorrectly classified as members. The score is the True Positive Rate achieved by this threshold (i.e., the proportion of correctly classified member challenge points). The live scoreboard shows additional scores (i.e., TPR at other FPRs, membership inference advantage, accuracy, AUC-ROC score). These are only informational.

You are allowed to make multiple submissions, but only your latest submission will be considered.


## Winner Selection

Winners will be selected independently for each task (i.e. if you choose not to participate in certain tasks, this will not affect your rank for the tasks in which you do participate). For each task, the winner will be the one achieving the highest score (`TPR @ 0.1 FPR`).
