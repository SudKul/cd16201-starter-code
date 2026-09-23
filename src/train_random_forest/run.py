#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

from components.compare_runs import effective_parameters, load_split
from components.local_pipeline import (
    load_artifact, read_json, relative_path, resolve_path,
    run_output, stage, write_json,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def fit_model(model, X_train, y_train):
    # YOUR CODE HERE: fit the inference pipeline using the training partition.
    raise NotImplementedError("Implement fitting in fit_model before training")


def export_model(model, model_dir):
    # YOUR CODE HERE: save the fitted pipeline in MLflow sklearn format at model_dir.
    raise NotImplementedError("Implement MLflow model export in export_model")


def log_validation_mae(mae):
    # YOUR CODE HERE: record mae in the active MLflow run under the key 'mae'.
    raise NotImplementedError("Implement validation MAE logging in log_validation_mae")


def categorical_preprocessor():
    # YOUR CODE HERE: impute the most frequent category, then one-hot encode it.
    raise NotImplementedError("Implement non-ordinal categorical preprocessing")


def assemble_pipeline(preprocessor, random_forest):
    # YOUR CODE HERE: assemble named 'preprocessor' and 'random_forest' steps.
    raise NotImplementedError("Implement inference pipeline assembly")


def configure_tracking(root):
    """Resolve the local backend explicitly, including when launched by MLflow."""
    tracking = root / "artifacts" / "mlflow"
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(tracking / "mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("nyc_airbnb")
    if experiment is None:
        experiment_id = client.create_experiment(
            "nyc_airbnb", artifact_location=(tracking / "artifacts").as_uri()
        )
    else:
        expected_location = (tracking / "artifacts").as_uri()
        if experiment.artifact_location != expected_location:
            raise ValueError("MLflow experiment artifact storage must use this project root")
        experiment_id = experiment.experiment_id
    return client, experiment_id


def go(args):
    root = Path(args.project_root).resolve()
    with stage(root, args.run_dir, "train_random_forest", inputs={
        "trainval": args.trainval, "split": args.split, "rf_config": args.rf_config,
    }) as record:
        split_path = resolve_path(root, args.split, must_exist=True)
        split = load_split(root, split_path)
        trainval = load_artifact(root, split["trainval"])
        if trainval != resolve_path(root, args.trainval, must_exist=True):
            raise ValueError("Training input does not match split metadata")
        load_artifact(root, split["test"])
        load_artifact(root, split["source"])
        if not 0 < args.val_size < 1:
            raise ValueError("val_size must be strictly between zero and one")
        output_dir = run_output(root, args.run_dir, args.output_dir)
        output_dir.mkdir(parents=True)
        rf_config = read_json(resolve_path(root, args.rf_config, must_exist=True))
        rf_config["random_state"] = args.random_seed
        parameters = effective_parameters({
            "random_forest": rf_config, "val_size": args.val_size,
            "random_seed": args.random_seed, "stratify_by": args.stratify_by,
            "max_tfidf_features": args.max_tfidf_features,
        })
        rf_config = parameters["random_forest"]
        write_json(output_dir / "parameters.json", parameters)
        X = pd.read_csv(trainval)
        y = X.pop("price")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.val_size,
            stratify=None if args.stratify_by == "none" else X[args.stratify_by],
            random_state=args.random_seed,
        )
        client, experiment_id = configure_tracking(root)
        # An explicit ID avoids accidentally resuming an outer project run.
        mlflow_run = client.create_run(experiment_id)
        with mlflow.start_run(run_id=mlflow_run.info.run_id):
            mlflow.set_tags({"pipeline_run_id": Path(args.run_dir).name,
                             "split_id": split["split_id"]})
            logged_parameters = {**rf_config, **{
                key: value for key, value in parameters.items() if key != "random_forest"
            }}
            mlflow.log_params(logged_parameters)
            sk_pipe, processed_features = get_inference_pipeline(
                rf_config, args.max_tfidf_features
            )
            fit_model(sk_pipe, X_train, y_train)
            r_squared = float(sk_pipe.score(X_val, y_val))
            mae = float(mean_absolute_error(y_val, sk_pipe.predict(X_val)))
            if not np.isfinite([mae, r_squared]).all():
                raise ValueError("Validation metrics must be finite")
            model_dir = output_dir / "model"
            export_model(sk_pipe, model_dir)
            if not (model_dir / "MLmodel").is_file():
                raise ValueError("Model export did not produce an MLflow model at model_dir")
            log_validation_mae(mae)
            mlflow.log_metric("r2", r_squared)
            fig = plot_feature_importance(sk_pipe, processed_features)
            fig.savefig(output_dir / "feature_importance.png")
            plt.close(fig)
            metrics = {"mae": mae, "r2": r_squared}
            write_json(output_dir / "metrics.json", metrics)
            for name in ("parameters.json", "metrics.json", "feature_importance.png"):
                mlflow.log_artifact(str(output_dir / name))
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            record["outputs"] = {
                "model": model_dir, "parameters": output_dir / "parameters.json",
                "metrics": output_dir / "metrics.json",
                "feature_importance": output_dir / "feature_importance.png",
            }
            record["details"] = {
                "mlflow_run_id": mlflow_run.info.run_id, "split_id": split["split_id"],
                "split": relative_path(root, split_path), "parameters": parameters,
                "metrics": metrics,
            }


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.asarray(np.append(feat_imp, nlp_importance))  # Using np.asarray for future compatibility
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10), layout='constrained')  # Using constrained layout
    sub_feat_imp.bar(np.arange(feat_imp.shape[0]), feat_imp, color="r", align="center")
    sub_feat_imp.set_xticks(np.arange(feat_imp.shape[0]))
    sub_feat_imp.set_xticklabels(feat_names, rotation=90)
    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):
    # Let's handle the categorical features first
    # Ordinal categorical are categorical values for which the order is meaningful, for example
    # for room type: 'Entire home/apt' > 'Private room' > 'Shared room'
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in production
    # (nor during training). That is not true for neighbourhood_group
    ordinal_categorical_preproc = OrdinalEncoder()

    ######################################
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = categorical_preprocessor()
    ######################################

    # Let's impute the numerical columns to make sure we can handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(
        lambda x: x.squeeze(),
        validate=False,
        feature_names_out="one-to-one",
    )
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        FunctionTransformer(
            lambda x: x.squeeze(),
            validate=False,
            feature_names_out="one-to-one",
        ),
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words="english",
        ),
    )
    
    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    # Create random forest
    random_forest = RandomForestRegressor(**rf_config)

    ######################################
    # Create the inference pipeline. The pipeline must have 2 steps: a step called "preprocessor" applying the
    # ColumnTransformer instance that we saved in the `preprocessor` variable, and a step called "random_forest"
    # with the random forest instance that we just saved in the `random_forest` variable.
    # HINT: Use the explicit Pipeline constructor so you can assign the names to the steps, do not use make_pipeline
    sk_pipe = assemble_pipeline(preprocessor, random_forest)

    return sk_pipe, processed_features


def build_parser():
    parser = argparse.ArgumentParser(description="Train from a local, identified split")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--trainval", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rf_config", required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stratify_by", default="none")
    parser.add_argument("--max_tfidf_features", type=int, default=5)
    return parser


if __name__ == "__main__":
    go(build_parser().parse_args())
