"""`detect_target` -- find the target column and task type, or say so honestly.

THE CENTRAL RULE THIS MODULE ENFORCES
------------------------------------------
Never return a target column the caller didn't ask for unless the
evidence for it is strong AND clearly better than the next-best
candidate. When it isn't, return `needs_clarification` with the
candidates and the reasoning behind each score -- never a silent guess.
This is a heuristic, explainable scorer, not a model; every score comes
with a plain-English reason a human (or an LLM relaying this to a human)
can sanity-check.
"""

import pandas as pd

_NAME_HINTS = ("target", "label", "class", "outcome", "result", "churn", "survived", "diagnosis")
# Matched by substring (e.g. "churn" matches "churned"). "y" is deliberately
# NOT in that list and matched by EXACT name only below -- as a substring
# it would match almost any column ("salary", "category", "myfeature", ...).
_EXACT_NAME_HINTS = ("y",)
_MAX_CLASSIFICATION_CARDINALITY = 20
_ID_LIKE_UNIQUE_RATIO = 0.95


def detect_target(
    df: pd.DataFrame,
    target_column: str | None = None,
    confidence_threshold: float = 0.6,
    margin_threshold: float = 0.25,
) -> dict:
    """Resolve a target column + task type, or report why it can't be resolved.

    If `target_column` is given (e.g. already set on the Dataset row, or
    explicitly supplied by the user/LLM this call), it's used directly --
    the ONLY thing checked is that the column actually exists. No
    auto-detection scoring runs at all in that case; an explicit choice
    is not something this function second-guesses.
    """
    if target_column:
        if target_column not in df.columns:
            return {
                "status": "needs_clarification",
                "target_column": None,
                "task_type": None,
                "candidates": [],
                "reasoning": (
                    f"Specified target column '{target_column}' does not exist in this "
                    f"dataset. Available columns: {df.columns.tolist()}"
                ),
            }
        return {
            "status": "resolved",
            "target_column": target_column,
            "task_type": _infer_task_type(df[target_column]),
            "candidates": [],
            "reasoning": "Target column was explicitly specified; used as given.",
        }

    candidates = _score_candidates(df)

    if not candidates:
        return {
            "status": "needs_clarification",
            "target_column": None,
            "task_type": None,
            "candidates": [],
            "reasoning": (
                "No column looks like a plausible target: every column is either "
                "constant, or has too many unique values to be a label (looks like "
                "an identifier)."
            ),
        }

    top = candidates[0]
    runner_up_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    confident = top["score"] >= confidence_threshold and (top["score"] - runner_up_score) >= margin_threshold

    if confident:
        return {
            "status": "resolved",
            "target_column": top["column"],
            "task_type": _infer_task_type(df[top["column"]]),
            "candidates": candidates[:5],
            "reasoning": f"'{top['column']}' scored clearly higher than other candidates: {top['reasoning']}.",
        }

    return {
        "status": "needs_clarification",
        "target_column": None,
        "task_type": None,
        "candidates": candidates[:5],
        "reasoning": (
            "No single column was a confident, clear-margin match for a target "
            "column -- ask the user which column to use."
        ),
    }


def _infer_task_type(series: pd.Series) -> str:
    """Numeric with many distinct values -> regression; everything else
    (text, or numeric with few distinct values, e.g. a 0/1-coded label)
    -> classification."""
    non_null = series.dropna()
    is_numeric = pd.api.types.is_numeric_dtype(non_null)
    n_unique = non_null.nunique()
    if is_numeric and n_unique > _MAX_CLASSIFICATION_CARDINALITY:
        return "regression"
    return "classification"


def _score_candidates(df: pd.DataFrame) -> list[dict]:
    n_rows = len(df)
    scored = []

    for i, col in enumerate(df.columns):
        series = df[col]
        n_unique = series.nunique(dropna=True)
        non_null_ratio = float(series.notna().mean()) if n_rows else 0.0

        if n_unique <= 1:
            continue  # constant column, can't be a useful target
        if n_rows and n_unique / n_rows > _ID_LIKE_UNIQUE_RATIO:
            continue  # looks like an identifier, not a label

        score = 0.0
        reasons = []

        name_lower = str(col).strip().lower()
        matched_hint = next((hint for hint in _NAME_HINTS if hint in name_lower), None)
        if not matched_hint and name_lower in _EXACT_NAME_HINTS:
            matched_hint = name_lower
        if matched_hint:
            score += 0.5
            reasons.append(f"column name '{col}' matches common target-name pattern '{matched_hint}'")

        if i == len(df.columns) - 1:
            score += 0.2
            reasons.append("it is the last column (a common dataset convention)")

        if 2 <= n_unique <= _MAX_CLASSIFICATION_CARDINALITY:
            score += 0.2
            reasons.append(f"low cardinality ({n_unique} unique values), consistent with a classification label")

        score += 0.1 * non_null_ratio

        scored.append(
            {
                "column": col,
                "score": round(score, 3),
                "unique_values": int(n_unique),
                "reasoning": "; ".join(reasons) if reasons else "weak signal only (no name/cardinality match)",
            }
        )

    scored.sort(key=lambda candidate: candidate["score"], reverse=True)
    return scored
