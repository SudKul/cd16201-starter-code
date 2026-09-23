#!/usr/bin/env python
"""Evaluate an explicitly selected local model on its associated held-out data."""
import argparse
import os
from pathlib import Path
import sys

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from components.compare_runs import load_selection
from components.local_pipeline import load_artifact, run_output, stage, write_json


def go(args):
    root = Path(args.project_root).resolve()
    with stage(root, args.run_dir, "test_regression_model", inputs={
        "selection": args.selection,
    }) as record:
        selection = load_selection(root, args.selection)
        model_path = load_artifact(root, selection["model"])
        test_path = load_artifact(root, selection["test"])
        output = run_output(root, args.run_dir, args.output)
        X_test = pd.read_csv(test_path)
        y_test = X_test.pop("price")
        model = mlflow.sklearn.load_model(str(model_path))
        prediction = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError("Held-out metrics must be finite")
        evidence = {
            "schema_version": 1, "run_id": Path(args.run_dir).name,
            "model_run_id": selection["run_id"], "model_path": selection["model"]["path"],
            "split_id": selection["split_id"], "test": selection["test"], **metrics,
        }
        write_json(output, evidence)
        record["outputs"] = {"metrics": output}
        record["details"] = evidence
        return evidence


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    go(build_parser().parse_args())
