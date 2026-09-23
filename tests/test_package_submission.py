"""Archive validation using static infrastructure records, not learner models."""
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from components.compare_runs import compare, load_selection
from components.local_pipeline import (
    artifact_record, create_run, read_json, relative_path, stage, update_stage, write_json,
)
from components.package_submission import DATA_TESTS, package
from components.train_val_test_split.run import go as split
from components.version_run import run_version
from test_version_run import fixture_repo

REPOSITORY = Path(__file__).resolve().parents[1]


def record_validation(root, run, source, failed=False):
    report = run / 'validation/report.xml'
    report.parent.mkdir(parents=True)
    cases = ''.join(
        f'<testcase name="{name}">'
        + ('<failure message="generic boundary failure" />'
           if failed and name == 'test_proper_boundaries' else '')
        + '</testcase>' for name in sorted(DATA_TESTS)
    )
    report.write_text('<testsuites><testsuite>' + cases + '</testsuite></testsuites>')
    update_stage(root, run, 'data_check', 'failed' if failed else 'completed',
                 inputs={'current': source, 'reference': root / 'artifacts/reference/reference.csv'},
                 outputs={'report': report})


def record_training(root, run, split_run, count, mae):
    metadata = read_json(split_run / 'split/split.json')
    parameters = {'random_forest': {'n_estimators': count, 'random_state': 42},
                  'val_size': .2, 'random_seed': 42, 'stratify_by': 'none', 'max_tfidf_features': 5}
    metrics = {'mae': mae, 'r2': .5}
    with stage(root, run, 'train_random_forest', inputs={
        'split': split_run / 'split/split.json', 'trainval': split_run / 'split/trainval.csv',
    }) as record:
        model = run / 'training/model'
        model.mkdir(parents=True)
        (model / 'MLmodel').write_text('generic static model marker; not executable\n')
        write_json(run / 'training/parameters.json', parameters)
        write_json(run / 'training/metrics.json', metrics)
        (run / 'training/feature_importance.png').write_bytes(b'generic image marker')
        record['outputs'] = {'model': model, 'parameters': run / 'training/parameters.json',
                             'metrics': run / 'training/metrics.json',
                             'feature_importance': run / 'training/feature_importance.png'}
        record['details'] = {'parameters': parameters, 'metrics': metrics,
                             'split_id': metadata['split_id'],
                             'split': relative_path(root, split_run / 'split/split.json')}


def mock_version_entries(monkeypatch):
    real_run = subprocess.run
    def run(command, **kwargs):
        if len(command) > 1 and command[1] == 'main.py':
            checkout = Path(kwargs['cwd'])
            commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            initial = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', '1.0.0'], text=True).strip()
            current = create_run(checkout, {})
            with stage(checkout, current, 'download') as record:
                (current / 'raw.csv').write_text('id,value\n' + ''.join(f'{i},{i}\n' for i in range(20)))
                record['outputs']['raw'] = current / 'raw.csv'
            with stage(checkout, current, 'basic_cleaning') as record:
                shutil.copy2(current / 'raw.csv', current / 'clean.csv')
                record['outputs']['clean'] = current / 'clean.csv'
            failed = commit == initial
            record_validation(checkout, current, current / 'clean.csv', failed=failed)
            if not failed:
                split(SimpleNamespace(project_root=checkout, run_dir=current, input=current / 'clean.csv',
                                      output_dir=current / 'split', test_size=.2, random_seed=42, stratify_by='none'))
                record_training(checkout, current, current, 10, 1.0)
            return subprocess.CompletedProcess(command, 1 if failed else 0)
        return real_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)


def submission_fixture(tmp_path, monkeypatch):
    root = fixture_repo(tmp_path)
    shutil.copytree(REPOSITORY, root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.git', 'artifacts', '__pycache__', '.pytest_cache'))
    write_json(root / 'src/eda/eda.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
                                         'metadata': {}, 'cells': []})
    base = create_run(root, {})
    with stage(root, base, 'basic_cleaning') as record:
        pd.DataFrame({'id': range(20), 'value': range(20)}).to_csv(base / 'clean.csv', index=False)
        record['outputs']['clean'] = base / 'clean.csv'
    record_validation(root, base, base / 'clean.csv')
    split(SimpleNamespace(project_root=root, run_dir=base, input=base / 'clean.csv',
                          output_dir=base / 'split', test_size=.2, random_seed=42, stratify_by='none'))
    runs = []
    for count, mae in [(10, 3.0), (20, 2.0), (30, 1.0)]:
        run = create_run(root, {})
        record_training(root, run, base, count, mae)
        runs.append(run)
    selected = compare(root, runs, select=True)
    evaluation = create_run(root, {})
    with stage(root, evaluation, 'test_regression_model', inputs={
        'selection': root / 'artifacts/selected_model.json',
    }) as record:
        output = evaluation / 'evaluation/metrics.json'
        write_json(output, {'schema_version': 1, 'run_id': evaluation.name,
                            'model_run_id': selected['run_id'], 'model_path': selected['model']['path'],
                            'split_id': selected['split_id'], 'test': selected['test'], 'mae': 1.5, 'r2': .4})
        record['outputs']['metrics'] = output
    mock_version_entries(monkeypatch)
    assert run_version(root, '1.0.0', 'sample2.csv') == 1
    assert run_version(root, '1.0.1', 'sample2.csv') == 0
    return root, selected


def test_archive_excludes_unrequested_content_and_relocates(tmp_path, monkeypatch):
    root, selected = submission_fixture(tmp_path, monkeypatch)
    for path in ['.env', 'credentials.json', 'src/reference_solution.py',
                 'src/eda/.ipynb_checkpoints/cached.ipynb', '.venv/library.py']:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('excluded generic fixture marker\n')
    archive = package(root, 'submission.zip')
    relocated = tmp_path / 'unpacked'
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        assert 'src/eda/eda.ipynb' in names
        assert 'artifacts/versions/source.bundle' in names
        assert 'components/get_data/data/sample2.csv' in names
        assert not any('reference_solution' in name or 'credentials' in name or '.env' in name
                       or 'checkpoints' in name or '.venv' in name or name.startswith('.git/') for name in names)
        zipped.extractall(relocated)
    assert load_selection(relocated, 'artifacts/selected_model.json') == selected
    # Repeat all evidence checks using extracted paths and local version markers.
    assert package(relocated, 'verified.zip').is_file()
    assert run_version(relocated, '1.0.0', 'sample2.csv') == 1


def test_missing_evidence_prevents_ready_archive(tmp_path, monkeypatch):
    root, selected = submission_fixture(tmp_path, monkeypatch)
    (root / 'src/eda/eda.ipynb').unlink()
    with pytest.raises(FileNotFoundError, match='Required input is missing'):
        package(root, 'submission.zip')
    assert not (root / 'submission.zip').exists()


def test_changed_metrics_prevent_ready_archive(tmp_path, monkeypatch):
    root, selected = submission_fixture(tmp_path, monkeypatch)
    manifest = read_json(root / selected['manifest_path'])
    output = manifest['stages']['train_random_forest']['outputs']['metrics']['path']
    (root / output).write_text('{"mae": 0, "r2": 1}\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        package(root, 'submission.zip')
    assert not (root / 'submission.zip').exists()


@pytest.mark.parametrize('missing', ['validation', 'initial_failure', 'updated_success'])
def test_incomplete_workflow_prevents_ready_archive(tmp_path, monkeypatch, missing):
    root, selected = submission_fixture(tmp_path, monkeypatch)
    if missing == 'validation':
        split_metadata = read_json(root / selected['split']['path'])
        manifest_path = root / 'artifacts/runs' / split_metadata['source_run_id'] / 'manifest.json'
        manifest = read_json(manifest_path)
        del manifest['stages']['data_check']
        manifest_path.write_text(json.dumps(manifest))
        expected = 'missing successful data validation'
    else:
        versions_path = root / 'artifacts/versions/versions.json'
        versions = read_json(versions_path)
        version = '1.0.0' if missing == 'initial_failure' else '1.0.1'
        record = versions['versions'][version]['executions'][0]
        evidence_path = root / record['path']
        evidence = read_json(evidence_path)
        evidence['exit_status'] = 0 if missing == 'initial_failure' else 1
        evidence_path.write_text(json.dumps(evidence))
        versions['versions'][version]['executions'][0] = artifact_record(root, evidence_path)
        versions_path.write_text(json.dumps(versions))
        expected = 'missing the required sample2'
    with pytest.raises(ValueError, match=expected):
        package(root, 'submission.zip')
    assert not (root / 'submission.zip').exists()


@pytest.mark.parametrize('cache_dir', ['cache', 'caches', 'Cache', '.cache'])
def test_referenced_cache_artifact_prevents_ready_archive(tmp_path, monkeypatch, cache_dir):
    root, selected = submission_fixture(tmp_path, monkeypatch)
    manifest_path = root / selected['manifest_path']
    cache = manifest_path.parent / cache_dir / 'model_cache.bin'
    cache.parent.mkdir()
    cache.write_bytes(b'generic cache fixture')
    manifest = read_json(manifest_path)
    manifest['stages']['train_random_forest']['outputs']['cache'] = artifact_record(root, cache)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe artifact path cannot be packaged'):
        package(root, 'submission.zip')
    assert not (root / 'submission.zip').exists()


@pytest.mark.parametrize('snapshot_name', ['config', 'input'])
def test_version_snapshot_must_match_bundled_tag(tmp_path, monkeypatch, snapshot_name):
    root, selected = submission_fixture(tmp_path, monkeypatch)
    versions_path = root / 'artifacts/versions/versions.json'
    versions = read_json(versions_path)
    evidence_path = root / versions['versions']['1.0.0']['executions'][0]['path']
    evidence = read_json(evidence_path)
    snapshot = root / evidence[snapshot_name]['path']
    snapshot.write_text('unrelated generic snapshot bytes\n')
    evidence[snapshot_name] = artifact_record(root, snapshot)
    evidence_path.write_text(json.dumps(evidence))
    audit_manifest = evidence_path.parent / 'manifest.json'
    manifest = read_json(audit_manifest)
    manifest['stages']['version_run']['outputs']['evidence'] = artifact_record(root, evidence_path)
    audit_manifest.write_text(json.dumps(manifest))
    versions['versions']['1.0.0']['executions'][0] = artifact_record(root, evidence_path)
    versions_path.write_text(json.dumps(versions))
    with pytest.raises(ValueError, match=f'{snapshot_name} snapshot differs from bundled tagged source'):
        package(root, 'submission.zip')
    assert not (root / 'submission.zip').exists()
