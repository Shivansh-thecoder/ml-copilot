"""Overfitting/underfitting diagnostics from train-vs-validation score evidence.

Never an absolute claim from one score -- every result is phrased as
"possible_X", carries the actual evidence (both scores, spelled out), and
a suggested next action, not a verdict. Comparing a SINGLE train score to
a SINGLE validation score is inherently limited (see `evidence` always
including both raw numbers so a human/LLM can judge for themselves); this
is a heuristic screen, not a certification.
"""

from .evaluation import HIGHER_IS_BETTER


def diagnose_fit(
    train_score: float,
    validation_score: float,
    metric_name: str,
    overfit_gap_ratio: float = 0.10,
    poor_score_threshold: float = 0.6,
) -> dict:
    """Compare a training-set score to a validation/CV score on the SAME metric.

    `possible_overfitting`: training score notably better than
    validation (a relative gap, not absolute, so it applies whether the
    metric is 0-1 bounded like F1 or unbounded like RMSE).

    `possible_underfitting`: BOTH scores are poor -- only checked for
    metrics bounded in [0, 1] where "poor" has a fixed meaning
    (`poor_score_threshold`); for unbounded metrics (MAE/RMSE) this
    engine doesn't guess what "poor" means in an arbitrary target's
    units, so underfitting isn't flagged for those -- `no_strong_signal`
    is returned instead, honestly, rather than a fabricated threshold.
    """
    higher_is_better = metric_name in HIGHER_IS_BETTER
    bounded_zero_one = metric_name in HIGHER_IS_BETTER  # accuracy/f1/roc_auc/r2-like; r2 can exceed this range but rarely in practice

    if higher_is_better:
        gap = train_score - validation_score
        reference = max(abs(train_score), 1e-9)
    else:
        gap = validation_score - train_score  # lower error is better; positive gap = train had lower (better) error
        reference = max(abs(validation_score), 1e-9)

    evidence = f"train {metric_name}={train_score:.4f}, validation {metric_name}={validation_score:.4f}"

    if gap / reference > overfit_gap_ratio:
        return {
            "diagnostic": "possible_overfitting",
            "evidence": evidence,
            "train_score": round(float(train_score), 4),
            "validation_score": round(float(validation_score), 4),
            "possible_action": "consider regularization, a simpler model, more training data, or feature reduction",
        }

    if bounded_zero_one and train_score < poor_score_threshold and validation_score < poor_score_threshold:
        return {
            "diagnostic": "possible_underfitting",
            "evidence": evidence,
            "train_score": round(float(train_score), 4),
            "validation_score": round(float(validation_score), 4),
            "possible_action": "consider a more expressive model, additional/better features, or less regularization",
        }

    return {
        "diagnostic": "no_strong_signal",
        "evidence": evidence,
        "train_score": round(float(train_score), 4),
        "validation_score": round(float(validation_score), 4),
        "possible_action": "train/validation gap does not strongly suggest over- or under-fitting on this metric alone",
    }
