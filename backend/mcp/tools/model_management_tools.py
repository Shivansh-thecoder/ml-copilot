"""MCP tools: `save_model`, `get_experiment_results`.

`save_model` designates one specific candidate from a completed
`train_models`/`tune_model` run as "the" saved model for a dataset.
`get_experiment_results` covers both "what happened in run X" and
"what's the currently saved model for this dataset" -- folded into one
tool rather than a separate `load_model` tool, since both answer "what
do we know about this dataset's models" and are usually wanted together.
"""

from ml_engine.persistence import (
    get_saved_model_metadata,
    list_experiments,
    list_saved_models as _list_saved_models,
    save_best_model,
)


def save_model(dataset_id: str, run_id: str, model_name: str) -> dict:
    """Save a specific trained model (from a prior train_models/tune_model
    run) as THE saved model for this dataset. Overwrites any previously
    saved model for the same dataset -- the full history of runs remains
    available via get_experiment_results regardless.

    Args:
        dataset_id: The id of the dataset the model was trained on.
        run_id: The run_id returned by the train_models or tune_model call
            that produced this model.
        model_name: Which model from that run to save, e.g. "random_forest".
    """
    try:
        return save_best_model(run_id=run_id, model_name=model_name, dataset_id=dataset_id)
    except ValueError as exc:
        return {"error": "save_failed", "message": str(exc)}
    except FileNotFoundError as exc:
        return {"error": "artifact_missing", "message": str(exc)}


def get_experiment_results(dataset_id: str) -> dict:
    """Retrieve past training/tuning runs for a dataset, and its currently
    saved model (if any) via save_model.

    Args:
        dataset_id: The id of the dataset to look up experiment history for.
    """
    runs = list_experiments(dataset_id)
    saved = get_saved_model_metadata(dataset_id)
    return {
        "dataset_id": dataset_id,
        "run_count": len(runs),
        "runs": runs,
        "saved_model": saved,
    }


def list_saved_models() -> dict:
    """List every dataset that currently has a saved model, with the model
    name, task type, target column and test metrics for each.

    Use this to find out which datasets can be predicted on with
    `predict_with_model`.
    """
    models = _list_saved_models()
    return {"count": len(models), "models": models}
