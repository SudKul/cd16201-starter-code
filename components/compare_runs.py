"""Compare completed local training runs using validation metrics only."""
import argparse
import csv
import io
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from components.local_pipeline import (
    artifact_record, load_artifact, read_json, read_manifest, relative_path,
    require_output, resolve_path,
)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_split(root, path):
    path = resolve_path(root, path, must_exist=True)
    split = read_json(path)
    if split.get("schema_version") != 1 or not split.get("split_id"):
        raise ValueError("Invalid split metadata")
    producer = path.parent.parent
    if require_output(root, producer, "data_split", "split_metadata") != path:
        raise ValueError("Split metadata does not belong to its producing run")
    for name in ("source", "trainval", "test"):
        artifact = load_artifact(root, split[name])
        if name != "source" and require_output(root, producer, "data_split", name) != artifact:
            raise ValueError(f"Split {name} does not match its producing run")
    source_run = Path("artifacts/runs") / split["source_run_id"]
    if require_output(root, source_run, "basic_cleaning", "clean") != load_artifact(root, split["source"]):
        raise ValueError("Split source does not match its cleaning run")
    expected_id = hashlib.sha256(json.dumps({
        "source_sha256": split["source"]["sha256"], "parameters": split["parameters"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if split["split_id"] != expected_id:
        raise ValueError("Split identity does not match its data and parameters")
    return split


def effective_parameters(parameters):
    """Compare actual forest settings, including defaults omitted from JSON."""
    from sklearn.ensemble import RandomForestRegressor

    parameters = dict(parameters)
    forest = parameters.get("random_forest")
    if not isinstance(forest, dict):
        raise ValueError("Training parameters must include a random_forest configuration")
    required = ("val_size", "random_seed", "stratify_by", "max_tfidf_features")
    if any(name not in parameters for name in required):
        raise ValueError("Training parameters are missing validation or preprocessing settings")
    forest = dict(forest)
    forest.setdefault("random_state", parameters["random_seed"])
    if forest["random_state"] != parameters["random_seed"]:
        raise ValueError("Forest random_state must match the recorded random_seed")
    parameters["random_forest"] = RandomForestRegressor(**forest).get_params(deep=False)
    return parameters


def training_candidate(root, run_dir):
    run_dir = resolve_path(root, run_dir, must_exist=True)
    manifest = read_manifest(root, run_dir)
    entry = manifest["stages"].get("train_random_forest", {})
    if manifest.get("status") != "completed" or entry.get("status") != "completed":
        raise ValueError(f"Run {run_dir.name} has not completed training")
    paths = {name: require_output(root, run_dir, "train_random_forest", name)
             for name in ("model", "parameters", "metrics", "feature_importance")}
    if not (paths["model"] / "MLmodel").is_file():
        raise ValueError(f"Run {run_dir.name} is missing its MLflow model")
    details = entry["details"]
    split_path = load_artifact(root, entry["inputs"]["split"])
    if relative_path(root, split_path) != details["split"]:
        raise ValueError("Training split path does not match its recorded input")
    split = load_split(root, split_path)
    if split["split_id"] != details["split_id"]:
        raise ValueError("Training split identity mismatch")
    if entry["inputs"]["trainval"] != split["trainval"]:
        raise ValueError("Training data does not match the identified split")
    parameters, metrics = read_json(paths["parameters"]), read_json(paths["metrics"])
    if parameters != details["parameters"] or metrics != details["metrics"]:
        raise ValueError("Training records disagree with the manifest")
    parameters = effective_parameters(parameters)
    for key in ("mae", "r2"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Validation {key} must be finite")
    if metrics["mae"] < 0:
        raise ValueError("Validation MAE cannot be negative")
    return {
        "schema_version": 1, "run_id": manifest["run_id"],
        "manifest_path": relative_path(root, run_dir / "manifest.json"),
        "model": artifact_record(root, paths["model"]),
        "split_id": split["split_id"], "split": artifact_record(root, split_path),
        "trainval": split["trainval"], "test": split["test"],
        "validation": {key: metrics[key] for key in ("mae", "r2")},
        "parameters": parameters,
    }


def load_selection(root, selection_path):
    selection = read_json(resolve_path(root, selection_path, must_exist=True))
    manifest = resolve_path(root, selection["manifest_path"], must_exist=True)
    expected = training_candidate(root, manifest.parent)
    if selection != expected:
        raise ValueError("Selection disagrees with its model, training run or split")
    return selection


def compare(root, runs, select=False):
    candidates = [training_candidate(root, run) for run in runs]
    if len(candidates) < 3 or len({c["run_id"] for c in candidates}) != len(candidates):
        raise ValueError("Supply at least three distinct completed training runs")
    comparison_keys = {
        (c["split"]["path"], c["split"]["sha256"], c["split_id"],
         c["parameters"]["val_size"], c["parameters"]["random_seed"],
         c["parameters"]["stratify_by"])
        for c in candidates
    }
    if len(comparison_keys) != 1:
        raise ValueError("Runs must use the same split artifact and validation partition")
    configurations = {json.dumps(c["parameters"], sort_keys=True) for c in candidates}
    if len(configurations) < 3:
        raise ValueError("Comparison requires three distinct hyperparameter configurations")
    candidates.sort(key=lambda c: (c["validation"]["mae"], c["run_id"]))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "run_id", "model_path", "split_id", "split_path", "parameters", "mae", "r2",
    ])
    writer.writeheader()
    for c in candidates:
        writer.writerow({
            "run_id": c["run_id"], "model_path": c["model"]["path"],
            "split_id": c["split_id"], "split_path": c["split"]["path"],
            "parameters": json.dumps(c["parameters"], sort_keys=True), **c["validation"],
        })
    atomic_text(resolve_path(root, "artifacts/comparison.csv"), output.getvalue())
    if select:
        atomic_text(resolve_path(root, "artifacts/selected_model.json"),
                    json.dumps(candidates[0], indent=2, sort_keys=True, allow_nan=False) + "\n")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    result = compare(Path(args.project_root).resolve(), args.runs, args.select)
    print(f"Lowest validation MAE: {result['run_id']} ({result['validation']['mae']})")


if __name__ == "__main__":
    main()
