"""MCP tools: `get_model_schema`, `predict_with_model`.

The tool boundary, same as every other file in `tools/`: translate the
call's arguments, delegate to `ml_engine.prediction`, and turn each known
failure into a structured `{"error": ...}` result instead of letting an
exception cross the MCP boundary as an opaque protocol error. No
validation, coercion, or scikit-learn call happens here -- all of it
lives in the ML engine, so the HTTP layer and the LLM get identical
behaviour from the same code.
"""

from ml_engine.prediction import (
    InvalidFeatureValueError,
    MissingFeaturesError,
    NoSavedModelError,
    UnreadableModelError,
    describe_saved_model,
    predict_with_saved_model,
)


def get_model_schema(dataset_id: str) -> dict:
    """Describe the model currently saved for a dataset: which model it is,
    its task type and target column, and the exact feature columns it expects
    for a prediction (each with its kind -- numeric or categorical -- and, for
    categorical columns, the categories it was trained on).

    Call this before `predict_with_model` to find out what feature values to
    supply.

    Args:
        dataset_id: The id of a dataset that has a saved model.
    """
    try:
        return describe_saved_model(dataset_id)
    except NoSavedModelError as exc:
        return {"error": "no_saved_model", "message": str(exc)}
    except FileNotFoundError as exc:
        return {"error": "artifact_missing", "message": str(exc)}
    except UnreadableModelError as exc:
        return {"error": "unreadable_model", "message": str(exc)}


def predict_with_model(dataset_id: str, rows: list[dict]) -> dict:
    """Predict target values for new feature rows using the model saved for a
    dataset. Applies the exact preprocessing the model was trained with.

    Returns one prediction per row; for classification it also returns each
    class's probability and the confidence of the predicted class, when the
    model supports probability estimates.

    Args:
        dataset_id: The id of a dataset that has a saved model.
        rows: One or more rows to predict on, each a {feature_name: value}
            object covering every feature from `get_model_schema`.
    """
    try:
        return predict_with_saved_model(dataset_id, rows)
    except NoSavedModelError as exc:
        return {"error": "no_saved_model", "message": str(exc)}
    except FileNotFoundError as exc:
        return {"error": "artifact_missing", "message": str(exc)}
    except UnreadableModelError as exc:
        return {"error": "unreadable_model", "message": str(exc)}
    except MissingFeaturesError as exc:
        return {
            "error": "missing_features",
            "message": str(exc),
            "row": exc.row_index,
            "missing": exc.missing,
        }
    except InvalidFeatureValueError as exc:
        return {
            "error": "invalid_feature_value",
            "message": str(exc),
            "row": exc.row_index,
            "column": exc.column,
        }
    except ValueError as exc:
        return {"error": "invalid_request", "message": str(exc)}
