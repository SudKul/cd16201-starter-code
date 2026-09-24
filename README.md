# Build an ML Pipeline for Short-Term Rental Prices in NYC

Build a reusable rental-price pipeline in the Udacity Workspace using bundled data,
MLflow, Hydra and scikit-learn. Normal project execution uses local files and
preinstalled software. No external account, API key, hosted repository or tracking
service is required.

This is a starter, not a completed model. Implement the marked cleaning, validation,
preprocessing, model assembly, fitting/export, MAE logging and stage-integration
TODOs. They deliberately raise `NotImplementedError` until completed. Infrastructure
for paths, isolated runs, tracking, comparison and evaluation is supplied.

## Setup

The Workspace should already have the pinned environment. For environment preparation
on Linux with Python 3.12 available, run from the project root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip==26.2
python -m pip install -r requirements.txt
python -m pip check
```

Dependency installation can use network access during preparation; subsequent pipeline
runs use the installed environment. `requirements.txt` is the canonical pinned set.
Optional `ydata-profiling` is omitted because its current release still imports
`pkg_resources`, which is absent from the patched setuptools version. Use pandas
and Matplotlib for EDA in this environment.
The optional Conda files refer to that same set; Conda-based provisioning is not part
of the verified setup. Run all commands below from the project root with this environment
active. Before any MLflow command, configure local tracking and disable telemetry
in each terminal, including terminals used for EDA or the optional UI:

```sh
export MLFLOW_DISABLE_TELEMETRY=true
mkdir -p artifacts/mlflow
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/artifacts/mlflow/mlflow.db"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
```

MLflow reads the telemetry setting during import, before the outer CLI starts the
pipeline. Repeat these exports in every new terminal. Always pass
`--env-manager=local` to `mlflow run`. Nested stage calls use the same
preinstalled environment. Runtime configuration is in `config.yaml`; supply overrides
through `hydra_options`. Avoid changing package versions during the exercise.

## Explore and implement

Both datasets are provided under `components/get_data/data/`. Start Jupyter locally:

```sh
MLFLOW_DISABLE_TELEMETRY=true mlflow run src/eda --env-manager=local
```

Create and save `src/eda/eda.ipynb`. Examine `sample1.csv`, investigate its columns,
price distribution, missing values and dates, and explain your cleaning decisions.
Notebook paths should be relative to the project; from `src/eda/`, bundled data is at
`../../components/get_data/data/sample1.csv`. Use pandas and Matplotlib for profiling. Do not
store credentials or external download dependencies in the notebook.
Open only notebooks and model artifacts from sources you trust; executing notebooks
and loading serialized models can run code.

Complete `src/basic_cleaning/run.py` for parameterized price/date cleaning. The
scaffold supplies CSV reading/writing. Complete the row-count and price-range checks
in `src/data_check/test_data.py`; preserve the supplied schema, neighborhood,
geographic and reference-distribution checks. Complete the designated training
functions in `src/train_random_forest/run.py`. Finally, complete the five indicated stage-parameter functions in `main.py` using
the contracts below. The supplied dispatcher invokes them with `env_manager="local"`.
Do not replace the supplied feature transformations or change the held-out split
while comparing models. No Cookiecutter generation is required.

## Establish the baseline and split

After implementing cleaning and its integration, run only ingestion and cleaning:

```sh
mlflow run . --env-manager=local -P steps="download,basic_cleaning"
```

Each invocation prints a new run directory. Set `R` to that exact project-relative
path, then initialize the reference once:

```sh
R=artifacts/runs/<cleaning-run-id>
python -m components.init_reference --project_root . --source "$R/clean.csv" --source_run "$R"
mlflow run . --env-manager=local -P steps="data_check,data_split" -P hydra_options="main.source_run=$R"
```

The reference is a fixed snapshot. Missing references stop validation; existing
references cannot be overwritten by initialization or later cleaning. The second
command sequence creates a new run containing validation evidence and the split.
Set `S` to this new run directory for all three training experiments below. Partial
runs require an explicit source; they never search for the newest artifact. A failed
validation exits nonzero and prevents downstream splitting and training.

Once the TODOs and reference are ready, a full default run is:

```sh
mlflow run . --env-manager=local
```

This runs ingestion through training. It does not select a model or evaluate the
held-out test set. An untouched starter is expected to stop at its first TODO.
For direct component commands and output filenames, see the interface contract below.

## Compare three configurations and evaluate

Use three distinct hyperparameter configurations with the same split and validation
partition. For example, vary the tree count while retaining the configured seed:

```sh
S=artifacts/runs/<validated-split-run-id>
mlflow run . --env-manager=local -P steps=train_random_forest -P hydra_options="main.source_run=$S modeling.random_forest.n_estimators=50"
mlflow run . --env-manager=local -P steps=train_random_forest -P hydra_options="main.source_run=$S modeling.random_forest.n_estimators=100"
mlflow run . --env-manager=local -P steps=train_random_forest -P hydra_options="main.source_run=$S modeling.random_forest.n_estimators=150"
```

Set `A`, `B` and `C` to these three printed run directories. Compare validation
metrics, then explicitly select the minimum validation MAE:

```sh
A=artifacts/runs/<first-training-run-id>
B=artifacts/runs/<second-training-run-id>
C=artifacts/runs/<third-training-run-id>
python -m components.compare_runs --project_root . --runs "$A" "$B" "$C"
python -m components.compare_runs --project_root . --runs "$A" "$B" "$C" --select
mlflow run . --env-manager=local -P steps=test_regression_model
```

Inspect `artifacts/comparison.csv`, `artifacts/selected_model.json`, the selected
run's `training/` files and the evaluation run's `evaluation/metrics.json`. Test data
is reserved for this final evaluation and cannot determine model selection. New
training runs leave the selection unchanged. The evaluator verifies the selected
model belongs to the recorded run and uses its matching held-out dataset.

An optional local UI can display the same tracking records:

```sh
MLFLOW_DISABLE_TELEMETRY=true MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1
```

The UI is not required for grading. JSON, CSV, plots and manifests provide the evidence.

## Local versions and new data

After completing and verifying your initial learner implementation, commit it in
your own local learner repository and create tag `1.0.0` using `git tag 1.0.0`.
If your learner copy has no Git history, initialize a local repository and commit the
starter source before recording these versions. Include the intended
hyperparameters in `config.yaml`. No remote push or hosted release is required.
Replay that exact tagged code against the second bundled sample:

```sh
python -m components.version_run --project_root . --version 1.0.0 --sample sample2.csv
```

The initial version should fail the supplied geographical-boundary validation and
stop before training. Inspect the recorded failure and add the required geographic
cleaning to your learner implementation. Verify it, commit the update locally and
create tag `1.0.1` using `git tag 1.0.1`, then run:

```sh
python -m components.version_run --project_root . --version 1.0.1 --sample sample2.csv
```

The helper executes existing tags in temporary isolated checkouts without changing
your working tree or fetching remote code. It records the exact commit, configuration,
input/reference checksums, exit status and run manifests under `artifacts/versions/`
and `artifacts/runs/`. The source bundle preserves the local markers for replay.
Keep both failed-initial and successful-updated evidence. Do not move an existing tag
to another revision. Keep the fixed reference from the original sample.

## Prepare the submission

Workspace is the recommended submission route. Save your work in the project
directory and use the course's Workspace submission option. This route does not
require a ZIP, GitHub account, hosted repository, public URL or remote push.

Every route requires the same grading evidence: source and configuration, bundled
data, the saved EDA notebook, selected model, comparison, explicit selection, final
evaluation, fixed reference, run manifests and both versioned-run records. Keep the
local Git tags and `artifacts/versions/source.bundle` for version replay. Use the
manifests and pipeline diagram to explain lineage; the tracking UI is optional.

GitHub and ZIP are optional alternatives in the course submission options. If you
choose GitHub, ensure reviewers can access the required evidence as well as source;
`artifacts/` is ignored by Git, so pushing source alone does not include that evidence.
If you choose ZIP, create the reviewer archive:

```sh
python -m components.package_submission --project_root . --output submission.zip
```

The packager checks required evidence and follows referenced artifacts. It includes
source, data, notebook, selected model, comparison/evaluation, manifests and local
version evidence. Credentials, virtual environments, caches, `.git` and the optional
MLflow database are excluded. Correct missing evidence before retrying a rejected
package. Inspect the archive and extract it into a new directory before submission;
activate the same pinned environment and repeat the setup of the local tracking URI
from that new root. Selection and held-out evaluation must resolve without original
absolute paths or external services. Version replay can use the included Git bundle
from the extracted directory without a `.git` directory; keep local Git history in
your working project.

The files collected by Workspace and replay of its actual submitted payload still
require verification in [#24](https://github.com/ruddyscent/cd16201-starter-code/issues/24).
ZIP packaging checks do not establish Workspace collection behavior, including
whether ignored files or local Git history are retained.

Maintainers must never add or publish completed reference answers in this starter,
its history, tests, attachments or release artifacts. Reference verification uses
temporary areas outside all repository worktrees; only solution-free evidence is
retained. Learner submissions contain the learner's own work. Workspace/course release
and full offline/relocation verification are separate from the focused component checks.

## Local interface contract

**Paths and runs.** All stored paths are relative to the project root, using `/`.
Resolve them against `--project_root`, never the process working directory; reject
paths outside that root. Absolute CLI paths are accepted only inside the root.
Every pipeline invocation creates `artifacts/runs/<run_id>/` with a unique ID;
components in that invocation share its `manifest.json`. A completed or failed
stage is never rerun into the same output directory: create another run and pass
its prerequisites explicitly. There is no `latest` alias or automatic selection.
CSV outputs have no index column. Normal execution uses bundled files and the
preinstalled environment, with no network requests or credentials.

**Manifest.** JSON schema version 1 has `schema_version`, `run_id`, `created_at`,
`status`, `config` (effective configuration), `code` (`commit`, `dirty`), and
`stages`. Each stage records `status` (`running`, `completed`, or `failed`),
`inputs`, `outputs`, and `details`; artifact entries contain `path` and `sha256`.
`details` includes stage parameters, source run/split identities where applicable,
metrics and failure details. The run status has the same three values. Persist
failure before propagating a nonzero exit status; incomplete runs cannot be model
candidates. A dirty or unavailable Git revision must be recorded honestly.
Checksums cover bytes; model-directory checksums use sorted relative filenames and
file checksums. Previously completed outputs must not be modified.

**Stage CLI.** The following are direct commands from the project root. Here `R`
is one freshly created run directory, such as `artifacts/runs/<run_id>`, and `S`
is the explicitly chosen source run. Infrastructure creates `R/manifest.json`;
`main.py` does this automatically. Python components take `--project_root .` and
`--run_dir "$R"`; their MLproject parameter names are the same without `--`.
The existing step names remain stable.

| Stage / owner | Example arguments (after script path) | Outputs within R |
| --- | --- | --- |
| `download` / #6: `components/get_data/run.py` | `--sample sample1.csv --output "$R/raw.csv"` | `raw.csv` copied from `components/get_data/data/` |
| `basic_cleaning` / #7: `src/basic_cleaning/run.py` | `--input "$R/raw.csv" --output "$R/clean.csv" --min_price 10 --max_price 350` | `clean.csv`, after learner implementation |
| `data_check` / #8: `python -m pytest src/data_check` | `--csv "$R/clean.csv" --ref artifacts/reference/reference.csv --kl_threshold 0.2 --min_price 10 --max_price 350 --junitxml "$R/validation/report.xml"` | `validation/report.xml`; orchestrator records exit status and input hashes |
| `data_split` / #9: `components/train_val_test_split/run.py` | `--input "$R/clean.csv" --output_dir "$R/split" --test_size 0.2 --random_seed 42 --stratify_by neighbourhood_group` | `split/trainval.csv`, `split/test.csv`, `split/split.json` |
| `train_random_forest` / #10–11: `src/train_random_forest/run.py` | `--trainval "$S/split/trainval.csv" --split "$S/split/split.json" --output_dir "$R/training" --val_size 0.2 --random_seed 42 --stratify_by neighbourhood_group --rf_config "$R/rf_config.json" --max_tfidf_features 5` | `training/model/`, `parameters.json`, `metrics.json`, `feature_importance.png` inside `training/` |
| `test_regression_model` / #13: `components/test_regression_model/run.py` | `--selection artifacts/selected_model.json --output "$R/evaluation/metrics.json"` | `evaluation/metrics.json` |
| EDA / #18: `mlflow run src/eda --env-manager=local` | Read `components/get_data/data/sample1.csv`; save learner notebook under `src/eda/` | Learner notebook; no run identity required for interactive exploration |

The pytest stage uses its five existing required data options, plus optional
`--project_root` for root-relative file resolution; it does not take `--run_dir`.
Its MLproject also accepts `report` for the JUnit destination. Missing or malformed
options fail before reading data. `stratify_by=none` disables stratification in
both split and training. Split sizes are fractions strictly between zero and one.

**Reference and split identity.** First run only download/cleaning, then explicitly
initialize the baseline:

```sh
python -m components.init_reference --project_root . --source "$R/clean.csv" --source_run "$R"
```

This creates `artifacts/reference/reference.csv` and `reference.json`; existing
reference directories are rejected, never replaced. The source must be the
completed cleaning output of the stated run. The metadata has `schema_version`,
`source_run_id`, `source` and `reference` artifact records. Every later validation
requires that baseline; an absent baseline stops with initialization instructions.
A normal cleaning run cannot initialize or update it.

`split.json` contains `schema_version`, `split_id`, `source_run_id`, `source`,
`parameters` (`test_size`, `random_seed`, `stratify_by`), `trainval`, and `test`.
The three dataset fields are artifact records. `split_id` is SHA-256 of canonical
JSON containing the source checksum and split parameters; the producer run is
recorded separately. Training validates the trainval checksum against this file,
records its project-relative path, and repeats the configured train/validation
split using its fixed seed. Comparison requires the exact same split artifact
and validation partition settings (`val_size`, `random_seed`, `stratify_by`), so
recreated but unrelated run artifacts are not silently substituted.

**Tracking, comparison and evaluation.** Configure MLflow at runtime with the
absolute URI for `<project_root>/artifacts/mlflow/mlflow.db` (SQLite), and local
file artifact storage at `<project_root>/artifacts/mlflow/artifacts/`. Never use
ambient remote tracking settings. The outer invocation must resolve this same
tracking context and export `MLFLOW_DISABLE_TELEMETRY=true` before starting MLflow;
nested calls inherit these settings and use `env_manager="local"`.
Portable JSON/CSV and model files are grading evidence; the tracking database/UI
is optional and is not relied on after relocation. Training logs effective forest
parameters (including resolved estimator defaults), preprocessing/partition settings,
validation `mae` and `r2`, and the
feature-importance image to one local MLflow run. Its manifest details link
`mlflow_run_id`, `split_id`, `split` (relative metadata path), `parameters`, and
`metrics`. `parameters.json` contains these effective modeling parameters;
`metrics.json` contains validation `mae` and `r2`. The learner retains the MAE
logging TODO; infrastructure provides all file/run plumbing.

```sh
python -m components.compare_runs --project_root . --runs "$A" "$B" "$C"
python -m components.compare_runs --project_root . --runs "$A" "$B" "$C" --select
```

The helper writes `artifacts/comparison.csv` from at least three completed runs
with distinct effective hyperparameter configurations and the same split and
validation partition. It verifies required model/split/metric files and finite
metrics. Columns are `run_id`, `model_path`, `split_id`, `split_path`,
`parameters` (JSON), `mae`, and `r2`. Explicit `--select` chooses minimum validation
MAE, breaking ties by run ID, and atomically writes `artifacts/selected_model.json`.
Without that flag neither comparison nor training changes the selection. Selection
fields are `schema_version`, `run_id`, `manifest_path`, `model` (artifact record),
`split_id`, `split` (artifact record), `trainval`, `test` (artifact records),
`validation` (`mae`, `r2`), and `parameters`. Evaluation accepts only this selection,
checks its manifest/model/split/checksums agree, and uses that split's held-out
`test` CSV. Its JSON records `schema_version`, `run_id` (evaluation run),
`model_run_id`, `model_path`, `split_id`, `test` artifact record, `mae`, and `r2`.
It never changes selection or reads test metrics for hyperparameter comparison.

**Orchestration and partial runs.** Root MLproject keeps `steps` and
`hydra_options`. Run with `mlflow run . --env-manager=local`; the outer local
tracking URI and telemetry opt-out are set by the documented shell setup before
this command.
`main.steps=all` orders download, cleaning, validation, splitting and training;
held-out evaluation is explicit. Root `main.py` resolves paths from its own
project root, creates a new run, writes `rf_config.json`, and passes absolute
resolved paths to components. Configuration adds `main.source_run` (default
`null`) and `main.selection` (default `artifacts/selected_model.json`); input
sample and existing modeling/validation settings retain their names.

For `mlflow run . --env-manager=local -P steps=train_random_forest -P
hydra_options="main.source_run=artifacts/runs/<split-run>"`, use only that
completed run's split. Other partial runs similarly require explicit predecessor
outputs from `main.source_run` unless produced earlier in this invocation. Never
search for the newest file or overwrite a source run. Check prerequisite stages,
checksums, reference metadata and parameters before executing; validation failure
prevents splitting/training. Training-only runs require evidence of successful
validation for the source cleaned data. Evaluation uses `main.selection`, not
`main.source_run`. Unknown/duplicate/out-of-order stages and missing prerequisites
fail clearly. Infrastructure validates contracts and dispatches components; learners complete
the five stage-parameter integration functions in `main.py`.

**Learner boundaries and verification.** Retain cleaning (price/date handling,
then geographic recovery), row-count/price-range tests, categorical preprocessing,
model assembly, fitting, MLflow model export, validation MAE logging, and the five
stage integrations as actionable `NotImplementedError` TODOs. Provide local
input retrieval, persistent directories, parameter/R²/image tracking, model
artifact recording, failure handling and subprocess plumbing. The former W&B
retrieval/upload TODOs become provided local infrastructure. Existing supplied
schema, geography and reference-distribution tests and feature engineering remain.
An untouched starter is syntactically valid and must fail clearly at unfinished
learning tasks. EDA remains learner work; Cookiecutter becomes optional or removed.

Reference completions exist only in temporary verification directories outside
this repository and every worktree, applied there to a copy bound to the candidate
revision. Never add an answer, patch, generator, completed notebook or answer
fixture to repository files/history, issue attachments or distributed artifacts.
Publish only solution-free commands, hashes, versions and pass/fail evidence.
Within the same pinned environment, input bytes, code, seed and settings must
reproduce split membership exactly; predictions and metrics use `rtol=1e-10`,
`atol=1e-10`. Apply the same tolerance to predictions before/after model reload.
Different environments are recorded separately, not claimed bit-identical.

**Versions and submission.** `python -m components.version_run --project_root .
--version 1.0.0 --sample sample2.csv` runs an existing local tag in an isolated
checkout without fetching or altering the working tree. The learner creates
`1.0.0` and the later `1.0.1` after their own completed work. Record tag-to-commit,
configuration, input/reference hashes, command/exit status and output manifest;
initial sample2 evidence must show the geographic failure and no training, while
the updated version passes. Execution outputs/evidence are copied back under a
fresh project run with relative paths. Exact tagged source/config is preserved
through `artifacts/versions/versions.json` and `artifacts/versions/source.bundle`
containing the two local tags. The helper cannot silently label dirty source as a
released version or create tags in this repository during implementation work.

`python -m components.package_submission --project_root . --output submission.zip`
validates then packages an allowlisted source tree, learner EDA, bundled data,
fixed reference, the three compared runs and their prerequisite/evaluation runs,
selection/comparison, manifests and version evidence. Include all transitively
referenced local artifacts and exclude credentials, `.git`, environments, caches,
tracking databases and separate reference solutions. Missing required evidence
must prevent a ready ZIP. Inspect and rerun after extracting under another root;
use local Git bundle evidence for versioned replay. Developer verification ZIPs
use solution-free fixtures only; reference-completed source is never packaged or
published. Full offline completion, sample2 recovery, repeated/partial runs,
relocation and fresh-Workspace/course consistency are separate checks (#19–29).

## License

[License](LICENSE.txt)
