"""Run the local rental pipeline with explicit, persistent stage inputs."""
import os
import sys
from pathlib import Path

# MLflow starts telemetry at import; this local pipeline does not contact it.
os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from components.compare_runs import load_split
from components.init_reference import load_reference
from components.local_pipeline import (
    create_run, load_artifact, read_manifest, relative_path, require_output, resolve_path,
    run_output, update_stage, write_json,
)

ROOT = Path(__file__).resolve().parent
ORDER = (
    'download', 'basic_cleaning', 'data_check', 'data_split',
    'train_random_forest', 'test_regression_model',
)
DEFAULT_STEPS = ORDER[:-1]
PROJECTS = {
    'download': 'components/get_data',
    'basic_cleaning': 'src/basic_cleaning',
    'data_check': 'src/data_check',
    'data_split': 'components/train_val_test_split',
    'train_random_forest': 'src/train_random_forest',
    'test_regression_model': 'components/test_regression_model',
}


def parse_steps(value):
    if value == 'all':
        return DEFAULT_STEPS
    if not isinstance(value, str):
        raise ValueError('main.steps must be all or a comma-separated stage list')
    steps = tuple(item.strip() for item in value.split(','))
    if not steps or any(not item for item in steps):
        raise ValueError('main.steps must contain stage names without empty entries')
    unknown = [item for item in steps if item not in ORDER]
    if unknown:
        raise ValueError(f'Unknown pipeline stage: {unknown[0]}')
    if len(steps) != len(set(steps)):
        raise ValueError('Duplicate pipeline stages are not allowed')
    if list(steps) != sorted(steps, key=ORDER.index):
        raise ValueError('Pipeline stages must follow download, cleaning, validation, split, training, evaluation order')
    return steps


def configure_tracking(root):
    tracking = root / 'artifacts' / 'mlflow'
    tracking.mkdir(parents=True, exist_ok=True)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    uri = 'sqlite:///' + str(tracking / 'mlflow.db')
    os.environ['MLFLOW_TRACKING_URI'] = uri
    mlflow.set_tracking_uri(uri)
    return uri


def source_run(root, config):
    value = config.main.source_run
    if not value:
        raise ValueError('This partial run needs main.source_run=artifacts/runs/<run_id>')
    source = resolve_path(root, str(value), must_exist=True)
    if not source.is_dir():
        raise ValueError(f'main.source_run is not a run directory: {source}')
    read_manifest(root, source)
    return source


def validated_clean(root, run_dir, expected=None):
    manifest = read_manifest(root, run_dir)
    validation = manifest['stages'].get('data_check', {})
    if validation.get('status') != 'completed':
        raise ValueError(f'Run {run_dir.name} needs completed data_check before splitting or training')
    require_output(root, run_dir, 'data_check', 'report')
    inputs = validation.get('inputs', {})
    if 'current' not in inputs or 'reference' not in inputs:
        raise ValueError('Validation manifest is missing current/reference input identity')
    clean = load_artifact(root, inputs['current'])
    if require_output(root, clean.parent, 'basic_cleaning', 'clean') != clean:
        raise ValueError('Validated data is not a completed cleaning output')
    if expected is not None and clean != expected:
        raise ValueError('Validation input differs from the cleaned dataset being split')
    fixed = load_reference(root)
    if load_artifact(root, inputs['reference']) != fixed:
        raise ValueError('Validation used a different reference dataset')
    return clean


def split_inputs(root, run_dir, config):
    split_path = require_output(root, run_dir, 'data_split', 'split_metadata')
    split = load_split(root, split_path)
    clean = load_artifact(root, split['source'])
    manifest = read_manifest(root, run_dir)
    validation_run = run_dir
    if manifest['stages'].get('data_check', {}).get('status') != 'completed':
        predecessor = manifest['config']['main'].get('source_run')
        if not predecessor:
            raise ValueError('Split run is missing its explicit validation source run')
        validation_run = resolve_path(root, predecessor, must_exist=True)
    if validated_clean(root, validation_run, expected=clean) != clean:
        raise ValueError('Split input differs from the validated dataset')
    expected = {
        'test_size': float(config.modeling.test_size),
        'random_seed': int(config.modeling.random_seed),
        'stratify_by': str(config.modeling.stratify_by),
    }
    if split['parameters'] != expected:
        raise ValueError('Split parameters differ from the effective modeling configuration')
    return require_output(root, run_dir, 'data_split', 'trainval'), split_path


def run_local_project(root, stage_name, parameters, run_dir):
    if not isinstance(parameters, dict):
        raise TypeError(f'{stage_name} integration must return a parameter dictionary')
    if 'project_root' in parameters or 'run_dir' in parameters:
        raise ValueError('Project root and run directory are supplied by the pipeline')
    values = dict(parameters)
    values['project_root'] = str(root)
    if stage_name != 'data_check':
        values['run_dir'] = str(run_dir)
    project = resolve_path(root, PROJECTS[stage_name], must_exist=True)
    mlflow.run(str(project), entry_point='main', parameters=values, env_manager='local')


def basic_cleaning_parameters(config, raw, run_dir):
    raise NotImplementedError(
        'TODO: integrate basic_cleaning by returning its input, output, min_price and max_price parameters.'
    )


def data_check_parameters(config, clean, reference, run_dir):
    raise NotImplementedError(
        'TODO: integrate data_check by returning csv, ref, thresholds and report parameters.'
    )


def data_split_parameters(config, clean, run_dir):
    raise NotImplementedError(
        'TODO: integrate data_split by returning input, output_dir and split parameters.'
    )


def train_random_forest_parameters(config, trainval, split, rf_config, run_dir):
    raise NotImplementedError(
        'TODO: integrate train_random_forest by returning its local split, configuration and output parameters.'
    )


def test_regression_model_parameters(config, selection, run_dir):
    raise NotImplementedError(
        'TODO: integrate test_regression_model by returning selection and output parameters.'
    )


def run_validation(root, run_dir, clean, reference, parameters):
    report = run_output(root, run_dir, parameters['report'])
    if resolve_path(root, parameters['csv']) != clean or resolve_path(root, parameters['ref']) != reference:
        raise ValueError('Validation parameters must use the current clean CSV and fixed reference')
    inputs = {'current': clean, 'reference': reference}
    update_stage(root, run_dir, 'data_check', 'running')
    try:
        run_local_project(root, 'data_check', parameters, run_dir)
        if not report.is_file():
            raise FileNotFoundError(f'Data check did not produce its JUnit report: {report}')
        update_stage(root, run_dir, 'data_check', 'completed', inputs=inputs,
                     outputs={'report': report}, details={
                         name: parameters[name] for name in ('kl_threshold', 'min_price', 'max_price')
                     })
    except BaseException as error:
        update_stage(root, run_dir, 'data_check', 'failed', inputs=inputs,
                     outputs={'report': report} if report.is_file() else None,
                     details={'error': f'{type(error).__name__}: {error}'})
        raise


def execute(config):
    root = ROOT
    configure_tracking(root)
    effective = OmegaConf.to_container(config, resolve=True)
    for name in ('source_run', 'selection'):
        value = effective['main'].get(name)
        if value:
            effective['main'][name] = relative_path(root, value)
    run_dir = create_run(root, effective)
    print(f'Pipeline run: {run_dir.relative_to(root).as_posix()}', flush=True)
    try:
        steps = parse_steps(config.main.steps)
        raw = clean = None
        split_run = None
        for name in steps:
            if name == 'download':
                run_local_project(root, name, {
                    'sample': str(config.etl.sample), 'output': str(run_dir / 'raw.csv'),
                }, run_dir)
                raw = require_output(root, run_dir, 'download', 'raw')
                if raw != run_dir / 'raw.csv':
                    raise ValueError('Download must write raw.csv in this run')
            elif name == 'basic_cleaning':
                if raw is None:
                    raw = require_output(root, source_run(root, config), 'download', 'raw')
                parameters = basic_cleaning_parameters(config, raw, run_dir)
                run_local_project(root, name, parameters, run_dir)
                clean = require_output(root, run_dir, 'basic_cleaning', 'clean')
                if clean != run_dir / 'clean.csv':
                    raise ValueError('Cleaning must write clean.csv in this run')
            elif name == 'data_check':
                if clean is None:
                    clean = require_output(root, source_run(root, config), 'basic_cleaning', 'clean')
                reference = load_reference(root)
                parameters = data_check_parameters(config, clean, reference, run_dir)
                run_validation(root, run_dir, clean, reference, parameters)
            elif name == 'data_split':
                if clean is None:
                    clean = validated_clean(root, source_run(root, config))
                else:
                    validated_clean(root, run_dir, expected=clean)
                parameters = data_split_parameters(config, clean, run_dir)
                run_local_project(root, name, parameters, run_dir)
                split_run = run_dir
                split_inputs(root, split_run, config)
            elif name == 'train_random_forest':
                if split_run is None:
                    split_run = source_run(root, config)
                trainval, split_path = split_inputs(root, split_run, config)
                rf_config = run_output(root, run_dir, run_dir / 'rf_config.json')
                write_json(rf_config, OmegaConf.to_container(config.modeling.random_forest, resolve=True))
                parameters = train_random_forest_parameters(
                    config, trainval, split_path, rf_config, run_dir)
                run_local_project(root, name, parameters, run_dir)
                require_output(root, run_dir, 'train_random_forest', 'model')
            elif name == 'test_regression_model':
                selection = resolve_path(root, str(config.main.selection), must_exist=True)
                parameters = test_regression_model_parameters(config, selection, run_dir)
                run_local_project(root, name, parameters, run_dir)
                require_output(root, run_dir, 'test_regression_model', 'metrics')
    except BaseException as error:
        if read_manifest(root, run_dir)['status'] != 'failed':
            update_stage(root, run_dir, 'orchestration', 'failed',
                         details={'error': f'{type(error).__name__}: {error}'})
        raise
    return run_dir


@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):
    execute(config)


if __name__ == '__main__':
    go()
