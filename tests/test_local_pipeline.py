"""Infrastructure checks using generic text artifacts, never learner solutions."""
import shutil

import pytest

from components.local_pipeline import (
    artifact_record, create_run, load_artifact, read_manifest, relative_path,
    resolve_path, stage,
)


def test_runs_are_unique_and_completed_stage_cannot_be_replaced(tmp_path):
    first = create_run(tmp_path, {'seed': 42})
    second = create_run(tmp_path, {'seed': 42})
    assert first != second
    output = first / 'example.txt'
    with stage(tmp_path, first, 'example') as record:
        output.write_text('first run')
        record['outputs']['example'] = output
    with pytest.raises(ValueError, match='already completed'):
        with stage(tmp_path, first, 'example'):
            output.write_text('replacement')
    assert output.read_text() == 'first run'
    assert read_manifest(tmp_path, first)['status'] == 'completed'
    assert read_manifest(tmp_path, second)['status'] == 'running'


def test_relocation_different_working_directory_and_integrity(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    run = create_run(root, {'sample': 'generic'})
    with stage(root, run, 'example') as record:
        (run / 'example.txt').write_text('portable bytes')
        record['outputs']['example'] = run / 'example.txt'
    moved = tmp_path / 'relocated'
    shutil.copytree(root, moved)
    monkeypatch.chdir(tmp_path)
    manifest = read_manifest(moved, relative_path(root, run))
    artifact = manifest['stages']['example']['outputs']['example']
    assert not artifact['path'].startswith('/')
    relocated_file = load_artifact(moved, artifact)
    assert relocated_file.read_text() == 'portable bytes'
    relocated_file.write_text('changed bytes')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_artifact(moved, artifact)
    relocated_file.unlink()
    with pytest.raises(FileNotFoundError, match='Required input is missing'):
        load_artifact(moved, artifact)


def test_failed_and_missing_input_are_not_successful(tmp_path):
    run = create_run(tmp_path, {})
    with pytest.raises(FileNotFoundError, match='Required input is missing'):
        with stage(tmp_path, run, 'example', inputs={'input': 'missing.csv'}):
            pytest.fail('missing input should fail before component work')
    manifest = read_manifest(tmp_path, run)
    assert manifest['status'] == 'failed'
    assert manifest['stages']['example']['status'] == 'failed'
    assert not manifest['stages']['example']['outputs']


def test_paths_cannot_escape_project_and_outputs_belong_to_run(tmp_path):
    with pytest.raises(ValueError, match='inside project root'):
        resolve_path(tmp_path, '../outside.csv')
    run = create_run(tmp_path, {})
    other = tmp_path / 'outside-run.txt'
    other.write_text('generic artifact')
    with pytest.raises(ValueError, match='their run directory'):
        with stage(tmp_path, run, 'example') as record:
            record['outputs']['other'] = other
    assert read_manifest(tmp_path, run)['status'] == 'failed'


def test_model_directory_checksum_detects_changes(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'metadata.txt').write_text('generic model metadata')
    recorded = artifact_record(tmp_path, model)
    (model / 'extra.txt').write_text('additional bytes')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_artifact(tmp_path, recorded)


def test_absolute_input_records_are_persisted_relative(tmp_path):
    source = tmp_path / 'input.txt'
    source.write_text('generic input')
    recorded = artifact_record(tmp_path, source)
    recorded['path'] = str(source)
    run = create_run(tmp_path, {})
    with stage(tmp_path, run, 'example', inputs={'source': recorded}):
        pass
    persisted = read_manifest(tmp_path, run)['stages']['example']['inputs']['source']
    assert persisted['path'] == 'input.txt'
    assert persisted['sha256'] == recorded['sha256']
