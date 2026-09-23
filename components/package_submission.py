"""Validate local evidence and build an allowlisted, portable submission ZIP."""
import argparse
import csv
import json
import math
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from components.compare_runs import load_selection, training_candidate
from components.init_reference import load_reference
from components.local_pipeline import (
    artifact_record, load_artifact, read_json, read_manifest, relative_path, require_output, resolve_path,
)

ROOT_FILES = ('README.md', 'LICENSE.txt', 'MLproject', 'main.py', 'config.yaml',
              'environment.yml', 'conda.yml', 'requirements.txt')
HELPERS = ('launch.py', 'local_pipeline.py', 'init_reference.py', 'compare_runs.py', 'version_run.py',
           'package_submission.py', 'README.md', 'conda.yml')
COMPONENTS = ('get_data', 'train_val_test_split', 'test_regression_model')
SOURCES = ('basic_cleaning', 'train_random_forest', 'data_check', 'eda')


def source_files(root):
    required = [*ROOT_FILES, *[f'components/{name}' for name in HELPERS],
                'components/get_data/data/sample1.csv', 'components/get_data/data/sample2.csv',
                'src/eda/eda.ipynb', 'src/data_check/conftest.py', 'src/data_check/test_data.py',
                'src/train_random_forest/feature_engineering.py']
    for parent, names in [('components', COMPONENTS), ('src', SOURCES)]:
        for name in names:
            required.extend(f'{parent}/{name}/{file}' for file in ('MLproject', 'conda.yml'))
            if name not in {'data_check', 'eda'}:
                required.append(f'{parent}/{name}/run.py')
    for path in required:
        yield resolve_path(root, path, must_exist=True)
    for path in ('components/__init__.py', 'components/setup.py', 'launch.py'):
        candidate = resolve_path(root, path)
        if candidate.is_file():
            yield candidate


DATA_TESTS = {
    'test_column_names', 'test_neighborhood_names', 'test_proper_boundaries',
    'test_similar_neigh_distrib', 'test_row_count', 'test_price_range',
}


def _validation_report(root, run_dir, expected_source, geographic_failure=False):
    manifest = read_manifest(root, run_dir)
    validation = manifest['stages'].get('data_check', {})
    expected_status = 'failed' if geographic_failure else 'completed'
    if validation.get('status') != expected_status:
        raise ValueError(f'Run {Path(run_dir).name} needs {expected_status} data_check evidence')
    inputs = validation.get('inputs', {})
    if inputs.get('current') != expected_source:
        raise ValueError('Validation input does not match its cleaned dataset')
    load_artifact(root, expected_source)
    fixed = artifact_record(root, load_reference(root))
    if inputs.get('reference') != fixed:
        raise ValueError('Validation reference does not match the fixed snapshot')
    report = validation.get('outputs', {}).get('report')
    if not report:
        raise ValueError('Validation evidence is missing its JUnit report')
    report_path = load_artifact(root, report)
    try:
        cases = list(ET.parse(report_path).getroot().iter('testcase'))
    except ET.ParseError as error:
        raise ValueError('Validation JUnit report is malformed') from error
    names = [case.get('name') for case in cases]
    if not DATA_TESTS.issubset(names) or any(names.count(name) != 1 for name in DATA_TESTS):
        raise ValueError('Validation report must contain all six required data tests exactly once')
    if any(case.find('skipped') is not None for case in cases):
        raise ValueError('Validation report cannot skip required data tests')
    failed = {case.get('name') for case in cases
              if case.find('failure') is not None or case.find('error') is not None}
    expected_failures = {'test_proper_boundaries'} if geographic_failure else set()
    if failed != expected_failures:
        raise ValueError('Validation report does not show the required geographical failure'
                         if geographic_failure else 'Validation report contains failing data tests')
    return resolve_path(root, run_dir) / 'manifest.json'


def _source_validation(root, selection):
    split = read_json(load_artifact(root, selection['split']))
    for path in sorted((root / 'artifacts/runs').glob('*/manifest.json')):
        entry = read_manifest(root, path.parent)['stages'].get('data_check', {})
        if entry.get('status') == 'completed' and entry.get('inputs', {}).get('current') == split['source']:
            return _validation_report(root, path.parent, split['source'])
    raise ValueError('Selected split is missing successful data validation and its JUnit report')


def _comparison(root, selection):
    path = resolve_path(root, 'artifacts/comparison.csv', must_exist=True)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3 or len({row['run_id'] for row in rows}) != len(rows):
        raise ValueError('Submission needs a comparison of at least three distinct runs')
    candidates = []
    for row in rows:
        candidate = training_candidate(root, Path('artifacts/runs') / row['run_id'])
        expected = {
            'model_path': candidate['model']['path'], 'split_id': candidate['split_id'],
            'split_path': candidate['split']['path'],
        }
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('Comparison paths or identities disagree with training evidence')
        if json.loads(row['parameters']) != candidate['parameters']:
            raise ValueError('Comparison parameters disagree with training evidence')
        if any(float(row[key]) != candidate['validation'][key] for key in ('mae', 'r2')):
            raise ValueError('Comparison metrics disagree with training evidence')
        candidates.append(candidate)
    settings = {(c['split']['path'], c['split']['sha256'], c['parameters']['val_size'],
                 c['parameters']['random_seed'], c['parameters']['stratify_by']) for c in candidates}
    if len(settings) != 1 or len({json.dumps(c['parameters'], sort_keys=True) for c in candidates}) < 3:
        raise ValueError('Comparison needs one split/validation partition and three configurations')
    best = min(candidates, key=lambda c: (c['validation']['mae'], c['run_id']))
    if selection != best:
        raise ValueError('Selection does not match the lowest validation MAE in comparison')
    return [resolve_path(root, c['manifest_path']) for c in candidates]


def _evaluation(root, selection):
    for path in sorted((root / 'artifacts/runs').glob('*/manifest.json')):
        manifest = read_manifest(root, path.parent)
        entry = manifest['stages'].get('test_regression_model', {})
        if manifest['status'] != 'completed' or entry.get('status') != 'completed':
            continue
        metrics_path = require_output(root, path.parent, 'test_regression_model', 'metrics')
        metrics = read_json(metrics_path)
        if metrics.get('model_run_id') != selection['run_id']:
            continue
        if (metrics.get('split_id') != selection['split_id'] or metrics.get('test') != selection['test']
                or metrics.get('model_path') != selection['model']['path']):
            raise ValueError('Evaluation does not match the selected model and held-out split')
        recorded_selection = load_artifact(root, entry['inputs']['selection'])
        if read_json(recorded_selection) != selection:
            raise ValueError('Evaluation used a different selection')
        if any(isinstance(metrics.get(key), bool) or not isinstance(metrics.get(key), (int, float))
               or not math.isfinite(metrics[key]) for key in ('mae', 'r2')) or metrics['mae'] < 0:
            raise ValueError('Held-out metrics must be finite and MAE nonnegative')
        return path
    raise ValueError('Missing completed evaluation for the selected model and held-out split')


def _versions(root):
    path = resolve_path(root, 'artifacts/versions/versions.json', must_exist=True)
    metadata = read_json(path)
    versions = metadata.get('versions', {})
    if set(versions) != {'1.0.0', '1.0.1'} or len({v['commit'] for v in versions.values()}) != 2:
        raise ValueError('Version evidence requires distinct local 1.0.0 and 1.0.1 revisions')
    bundle = load_artifact(root, metadata['bundle'])
    heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True)
    refs = {line.split()[1]: line.split()[0] for line in heads.splitlines()}
    for version, entry in versions.items():
        # Annotated tags list their tag object; peel in a temporary repository below.
        if f'refs/tags/{version}' not in refs:
            raise ValueError(f'Version bundle is missing {version}')
        if not entry.get('executions'):
            raise ValueError(f'Version {version} has no recorded execution')
        matching_outcome = False
        for record in entry['executions']:
            evidence = read_json(load_artifact(root, record))
            if evidence['version'] != version or evidence['commit'] != entry['commit']:
                raise ValueError('Version execution disagrees with its recorded revision')
            if not evidence.get('manifests'):
                raise ValueError('Version execution is missing its pipeline manifest')
            for manifest_record in evidence['manifests']:
                manifest = read_json(load_artifact(root, manifest_record))
                if manifest['code'] != {'commit': entry['commit'], 'dirty': False}:
                    raise ValueError('Version output manifest does not identify clean tagged code')
                run_dir = resolve_path(root, manifest_record['path']).parent
                if evidence.get('sample') != 'sample2.csv':
                    continue
                if version == '1.0.0' and evidence.get('exit_status', 0) != 0:
                    stages = manifest['stages']
                    if manifest.get('status') != 'failed' or 'train_random_forest' in stages:
                        raise ValueError('Initial sample2 version must fail before training')
                    clean = require_output(root, run_dir, 'basic_cleaning', 'clean')
                    _validation_report(root, run_dir, artifact_record(root, clean), geographic_failure=True)
                    matching_outcome = True
                elif version == '1.0.1' and evidence.get('exit_status') == 0:
                    if not all(manifest['stages'].get(name, {}).get('status') == 'completed'
                               for name in ('download', 'basic_cleaning', 'data_check',
                                            'data_split', 'train_random_forest')):
                        raise ValueError('Updated sample2 version must complete every training pipeline stage')
                    candidate = training_candidate(root, run_dir)
                    _source_validation(root, candidate)
                    matching_outcome = True
        if not matching_outcome:
            raise ValueError(f'Version {version} is missing the required sample2 '
                             + ('geographical failure before training' if version == '1.0.0'
                                else 'successful completed pipeline execution'))
    import tempfile
    with tempfile.TemporaryDirectory(prefix='cd16201-bundle-check-') as temporary:
        subprocess.run(['git', 'clone', '--no-checkout', str(bundle), temporary],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for version, entry in versions.items():
            commit = subprocess.check_output(['git', '-C', temporary, 'rev-parse', f'{version}^{{commit}}'], text=True).strip()
            if commit != entry['commit']:
                raise ValueError('Version bundle disagrees with the recorded tag revision')
            for record in entry['executions']:
                evidence = read_json(load_artifact(root, record))
                sample = evidence.get('sample')
                if sample not in {'sample1.csv', 'sample2.csv'}:
                    raise ValueError('Version execution identifies an unknown bundled sample')
                for name, tagged_path in (
                    ('config', 'config.yaml'),
                    ('input', f'components/get_data/data/{sample}'),
                ):
                    snapshot = load_artifact(root, evidence[name])
                    tagged_bytes = subprocess.check_output(
                        ['git', '-C', temporary, 'show', f'{commit}:{tagged_path}'])
                    if snapshot.read_bytes() != tagged_bytes:
                        raise ValueError(f'Version {version} {name} snapshot differs from bundled tagged source')
    return path


def _artifact_closure(root, seeds):
    pending, included = list(seeds), set()
    while pending:
        path = resolve_path(root, pending.pop(), must_exist=True)
        if path in included:
            continue
        if path.is_symlink():
            raise ValueError(f'Submission artifacts cannot be symlinks: {path}')
        if path.is_dir():
            pending.extend(path.rglob('*'))
            continue
        included.add(path)
        relative = Path(relative_path(root, path))
        if relative.parts[:2] == ('artifacts', 'runs'):
            pending.append(root / Path(*relative.parts[:3]) / 'manifest.json')
        if path.suffix != '.json':
            continue
        def visit(value):
            if isinstance(value, dict):
                if 'path' in value and 'sha256' in value:
                    pending.append(load_artifact(root, value))
                if 'manifest_path' in value:
                    pending.append(resolve_path(root, value['manifest_path'], must_exist=True))
                if 'execution_run_id' in value:
                    pending.append(root / 'artifacts/runs' / value['execution_run_id'] / 'manifest.json')
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(read_json(path))
    return included


def package(project_root, output):
    root = Path(project_root).resolve()
    destination = resolve_path(root, output)
    if destination.exists():
        raise FileExistsError(f'Submission output already exists: {destination}')
    source = set(source_files(root))
    notebook = read_json(root / 'src/eda/eda.ipynb')
    if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
        raise ValueError('src/eda/eda.ipynb must be a readable Jupyter notebook')
    load_reference(root)
    selection_path = root / 'artifacts/selected_model.json'
    selection = load_selection(root, selection_path)
    seeds = [selection_path, root / 'artifacts/comparison.csv',
             root / 'artifacts/reference/reference.json', *_comparison(root, selection),
             _source_validation(root, selection), _evaluation(root, selection), _versions(root)]
    files = source | _artifact_closure(root, seeds)
    for path in files:
        relative = Path(relative_path(root, path))
        if path not in source and relative.parts[0] != 'artifacts':
            raise ValueError(f'Artifact references non-allowlisted source: {relative}')
        if any(part.startswith('.') or part in {'__pycache__', 'node_modules', 'venv', 'env'}
               or part.casefold() in {'cache', 'caches'}
               or 'secret' in part.lower() or 'credential' in part.lower()
               or 'reference_solution' in part.lower() for part in relative.parts):
            raise ValueError(f'Unsafe artifact path cannot be packaged: {relative}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, relative_path(root, path))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', default='submission.zip')
    args = parser.parse_args()
    print(package(args.project_root, args.output))


if __name__ == '__main__':
    main()
