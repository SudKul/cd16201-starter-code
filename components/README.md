# Local pipeline components

Use the preinstalled Python 3.12 environment, or prepare it from the pinned root
`requirements.txt` as described in [the project README](../README.md#setup).
Normal execution uses bundled data and project-local files.

Before starting any MLflow CLI, follow the root README shell setup: export
`MLFLOW_DISABLE_TELEMETRY=true` and the project-local `MLFLOW_TRACKING_URI`. Repeat
this in each new terminal, including EDA and optional UI terminals, because MLflow
initializes telemetry during import.

Run the root project with `mlflow run . --env-manager=local`. Components use the
same environment; their MLproject parameter names match their Python CLI options.
Use `--help` on a component's `run.py` to inspect its arguments. Component paths,
run manifests and outputs are specified in the
[local interface contract](../README.md#local-interface-contract).

The `local_pipeline` module provides path/checksum and run-status helpers.
`init_reference` creates the fixed baseline; `compare_runs` compares validation
metrics and selects a model explicitly; `version_run` replays existing local tags;
`package_submission` collects portable reviewer evidence. These modules run with
`python -m components.<module>` from the project root. No package installation of
this source tree or external tracking account is required.
