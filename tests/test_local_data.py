"""Subprocess checks of infrastructure using bundled or generic CSV data."""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from components.local_pipeline import (
    checksum, create_run, load_artifact, read_json, read_manifest, relative_path, stage,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def component(script, root, run, *arguments):
    return subprocess.run(
        [sys.executable, str(REPOSITORY / script), '--project_root', str(root),
         '--run_dir', relative_path(root, run), *arguments],
        cwd=root.parent, text=True, capture_output=True,
    )


def test_both_bundled_inputs_survive_repeated_ingestion(tmp_path):
    root = tmp_path / 'project'
    shutil.copytree(REPOSITORY / 'components/get_data/data', root / 'components/get_data/data')
    outputs = []
    for sample in ['sample1.csv', 'sample2.csv', 'sample1.csv']:
        run = create_run(root, {'sample': sample})
        output = run / 'raw.csv'
        result = component('components/get_data/run.py', root, run,
                           '--sample', sample, '--output', relative_path(root, output))
        assert result.returncode == 0, result.stderr
        assert checksum(output) == checksum(root / 'components/get_data/data' / sample)
        outputs.append(output)
    assert len(set(outputs)) == 3
    assert outputs[0].read_bytes() == outputs[2].read_bytes()
    assert read_manifest(root, outputs[0].parent)['stages']['download']['status'] == 'completed'
    run = create_run(root, {})
    missing = component('components/get_data/run.py', root, run,
                        '--sample', 'missing.csv', '--output', relative_path(root, run / 'raw.csv'))
    assert missing.returncode != 0
    assert 'Unknown bundled sample' in missing.stderr
    assert read_manifest(root, run)['status'] == 'failed'


@pytest.mark.parametrize('stratify_by', ['group', 'none'])
def test_split_membership_and_provenance_in_subsequent_process(tmp_path, stratify_by):
    root = tmp_path / 'project'
    source_run = create_run(root, {})
    source = source_run / 'clean.csv'
    with stage(root, source_run, 'basic_cleaning') as record:
        pd.DataFrame({'id': range(40), 'group': ['a', 'b'] * 20,
                      'value': range(100, 140)}).to_csv(source, index=False)
        record['outputs']['clean'] = source
    results = []
    for _ in range(2):
        run = create_run(root, {})
        split = run / 'split'
        result = component('components/train_val_test_split/run.py', root, run,
                           '--input', relative_path(root, source),
                           '--output_dir', relative_path(root, split),
                           '--test_size', '0.25', '--random_seed', '42', '--stratify_by', stratify_by)
        assert result.returncode == 0, result.stderr
        metadata = read_json(split / 'split.json')
        trainval = pd.read_csv(load_artifact(root, metadata['trainval']))
        test = pd.read_csv(load_artifact(root, metadata['test']))
        assert set(trainval.id).isdisjoint(test.id)
        assert set(trainval.id) | set(test.id) == set(range(40))
        assert len(test) == 10
        assert metadata['source_run_id'] == source_run.name
        assert load_artifact(root, metadata['source']) == source
        results.append((metadata, trainval.id.tolist(), test.id.tolist()))
    assert results[0][0]['split_id'] == results[1][0]['split_id']
    assert results[0][1:] == results[1][1:]
    # A separate reader uses persisted, relative artifact paths after relocation.
    moved = tmp_path / 'moved'
    shutil.copytree(root, moved)
    script = (
        'import json,pathlib,pandas as pd,sys; root=pathlib.Path(sys.argv[1]); '
        'm=json.loads((root/sys.argv[2]).read_text()); '
        'assert len(pd.read_csv(root/m["test"]["path"])) == 10'
    )
    reader = subprocess.run([sys.executable, '-c', script, str(moved),
                             relative_path(root, split / 'split.json')], cwd=tmp_path)
    assert reader.returncode == 0


def test_split_rejects_tampered_source(tmp_path):
    run = create_run(tmp_path, {})
    source = run / 'clean.csv'
    with stage(tmp_path, run, 'basic_cleaning') as record:
        source.write_text('id,value\n1,a\n2,b\n')
        record['outputs']['clean'] = source
    source.write_text('id,value\n1,changed\n2,b\n')
    result = component('components/train_val_test_split/run.py', tmp_path, run,
                       '--input', relative_path(tmp_path, source),
                       '--output_dir', relative_path(tmp_path, run / 'split'), '--test_size', '0.5')
    assert result.returncode != 0
    assert 'checksum mismatch' in result.stderr
    assert not (run / 'split').exists()
    assert read_manifest(tmp_path, run)['status'] == 'failed'
