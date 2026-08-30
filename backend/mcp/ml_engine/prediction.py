"""Predicting on new rows with a previously saved model.

WHY THIS RELOADS THE FITTED PIPELINE INSTEAD OF PREPROCESSING AGAIN
-------------------------------------------------------------------
A saved artifact is a fitted `Pipeline([("preprocessing", ...), ("model", ...)])`
(see persistence.py). Its preprocessing step is not just a recipe -- it
carries the numbers LEARNED from the training split: each numeric
column's imputation median and the mean/scale `StandardScaler` centred it
by, and the exact category list `OneHotEncoder` assigned column positions
to. Rebuilding a preprocessor here and fitting it on the incoming rows
would produce different medians, a different scale, and a different
one-hot column ORDER -- the estimator would then read feature 3 as
whatever landed in slot 3 this time. The prediction wouldn't error; it
would just be quietly wrong. So this module only ever calls `.predict()`
/ `.predict_proba()` on the loaded pipeline, and never fits anything.

The same artifact is also where the feature schema comes from: the
fitted pipeline knows which columns it was trained on, in what order,
and which of them were treated as numeric vs. categorical -- so nothing
here (or in the API/UI above it) needs to know anything about a
particular dataset's columns.
"""

import numpy as np
import pandas as pd

from .persistence import get_saved_model_metadata, load_saved_model


class NoSavedModelError(Exception):
    """No model has been saved for this dataset yet (see `save_model`)."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(
            f"No saved model for dataset '{dataset_id}'. Train models and save one first."
        )


class UnreadableModelError(Exception):
    """The artifact exists but isn't a pipeline this module can describe/use."""


class MissingFeaturesError(Exception):
    """One or more required feature columns weren't supplied for a row."""

    def __init__(self, row_index: int, missing: list[str]) -> None:
        self.row_index = row_index
        self.missing = missing
        super().__init__(
            f"Row {row_index} is missing required feature(s): {', '.join(missing)}."
        )


class InvalidFeatureValueError(Exception):
    """A supplied value isn't usable for the column it was given for."""

    def __init__(self, row_index: int, column: str, value, expected: str) -> None:
        self.row_index = row_index
        self.column = column
        self.value = value
        self.expected = expected
        super().__init__(
            f"Row {row_index}: value {value!r} for feature '{column}' is not {expected}."
        )


def describe_saved_model(dataset_id: str) -> dict:
    """The saved model's metadata plus the feature schema new rows must match."""
    metadata = get_saved_model_metadata(dataset_id)
    if metadata is None:
        raise NoSavedModelError(dataset_id)

    pipeline = load_saved_model(dataset_id)
    classes = _class_labels(pipeline)

    return {
        **metadata,
        "features": _feature_schema(pipeline),
        "classes": classes,
        "supports_probabilities": bool(classes) and hasattr(pipeline, "predict_proba"),
    }


def predict_with_saved_model(dataset_id: str, rows: list[dict]) -> dict:
    """Run new feature rows through this dataset's saved fitted pipeline."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("`rows` must be a non-empty list of {feature: value} objects.")

    metadata = get_saved_model_metadata(dataset_id)
    if metadata is None:
        raise NoSavedModelError(dataset_id)

    pipeline = load_saved_model(dataset_id)
    features = _feature_schema(pipeline)
    feature_names = [f["name"] for f in features]

    prepared, ignored, warnings = _prepare_rows(rows, features)
    # Column order matters as much as column presence: the fitted
    # ColumnTransformer selects by name, but the estimator behind it was
    # trained on positions. Building the frame in `feature_names_in_`
    # order is what keeps those two aligned.
    frame = pd.DataFrame(prepared, columns=feature_names)

    predicted = pipeline.predict(frame)
    classes = _class_labels(pipeline)
    probabilities = (
        pipeline.predict_proba(frame) if classes and hasattr(pipeline, "predict_proba") else None
    )

    results = []
    for i in range(len(frame)):
        entry = {"row": i, "prediction": _to_native(predicted[i])}
        if probabilities is not None:
            per_class = {str(c): float(p) for c, p in zip(classes, probabilities[i])}
            entry["probabilities"] = per_class
            entry["confidence"] = max(per_class.values())
        results.append(entry)

    return {
        "dataset_id": dataset_id,
        "model_name": metadata.get("model_name"),
        "task_type": metadata.get("task_type"),
        "target_column": metadata.get("target_column"),
        "classes": classes,
        "predictions": results,
        "ignored_columns": ignored,
        "warnings": warnings,
    }


def _feature_schema(pipeline) -> list[dict]:
    """Describe the columns the pipeline was fitted on, in training order.

    Everything here is read off the FITTED objects inside the artifact --
    `feature_names_in_` for the names and their order, the
    ColumnTransformer's fitted transformer list for which columns were
    treated as numeric vs. categorical, and the fitted OneHotEncoder's
    `categories_` for the values each categorical column actually saw
    during training.
    """
    names = list(getattr(pipeline, "feature_names_in_", []))
    if not names:
        raise UnreadableModelError(
            "This saved model doesn't record the feature columns it was trained on, "
            "so a prediction form can't be built for it. Retrain and save it again."
        )

    kinds, categories = _column_roles(pipeline)

    schema = []
    for name in names:
        kind = kinds.get(name, "unknown")
        entry = {"name": name, "kind": kind}
        if kind == "categorical" and name in categories:
            entry["categories"] = categories[name]
        schema.append(entry)
    return schema


def _column_roles(pipeline) -> tuple[dict, dict]:
    """{column: "numeric"|"categorical"} and {column: [known categories]}."""
    preprocessor = pipeline.named_steps.get("preprocessing") if hasattr(pipeline, "named_steps") else None
    kinds: dict[str, str] = {}
    categories: dict[str, list] = {}
    if preprocessor is None:
        return kinds, categories

    for name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if name not in ("numeric", "categorical") or transformer == "drop":
            continue
        for column in columns:
            kinds[column] = name
        if name == "categorical":
            encoder = getattr(transformer, "named_steps", {}).get("encode")
            for column, known in zip(columns, getattr(encoder, "categories_", [])):
                categories[column] = [_to_native(value) for value in known]

    return kinds, categories


def _prepare_rows(rows: list[dict], features: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """Validate and coerce raw input rows; return them plus what was ignored/warned about."""
    ignored: set[str] = set()
    warnings: list[str] = []
    prepared = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object of {{feature: value}} pairs.")

        missing = [f["name"] for f in features if _is_blank(row.get(f["name"]))]
        if missing:
            raise MissingFeaturesError(index, missing)

        ignored.update(set(row) - {f["name"] for f in features})

        values = {}
        for feature in features:
            name, kind = feature["name"], feature["kind"]
            raw = row[name]
            if kind == "numeric":
                values[name] = _as_number(index, name, raw)
            else:
                values[name] = str(raw)
                known = feature.get("categories")
                if known and values[name] not in {str(c) for c in known}:
                    # OneHotEncoder(handle_unknown="ignore") encodes an
                    # unseen category as all zeros rather than failing, so
                    # this is a usable-but-degraded prediction, not an error.
                    warnings.append(
                        f"Row {index}: '{raw}' is not a category '{name}' was trained on; "
                        "the model treats it as unknown."
                    )
        prepared.append(values)

    return prepared, sorted(ignored), warnings


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _as_number(row_index: int, column: str, value) -> float:
    if isinstance(value, bool):
        raise InvalidFeatureValueError(row_index, column, value, "a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidFeatureValueError(row_index, column, value, "a number") from None


def _class_labels(pipeline) -> list | None:
    """Class labels for a classifier, or None for a regressor."""
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        return None
    return [_to_native(c) for c in classes]


def _to_native(value):
    """numpy scalars -> plain Python, so results survive JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    return value
