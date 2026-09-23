"""Real local Git isolation with a mocked learner entry (no reference answers)."""
import shutil
import subprocess
from pathlib import Path

from components.init_reference import initialize_reference
from components.local_pipeline import create_run, read_json, read_manifest, stage
from components.version_run import git, run_version

REPOSITORY = Path(__file__).resolve().parents[1]


def fixture_repo(tmp_path):
    root = tmp_path / 'learner'
    subprocess.run(['git', 'clone', '--no-hardlinks', str(REPOSITORY), str(root)],
                   check=True, capture_output=True)
    commits = git(root, 'rev-list', '--max-count=2', 'HEAD').splitlines()
    assert len(commits) == 2
    for version, commit in zip(['1.0.0', '1.0.1'], reversed(commits)):
        subprocess.run(['git', '-C', str(root), 'update-ref', f'refs/tags/{version}', commit], check=True)
    baseline_run = create_run(root, {})
    with stage(root, baseline_run, 'basic_cleaning') as record:
        baseline = baseline_run / 'clean.csv'
        baseline.write_text('id,value\n1,generic\n')
        record['outputs']['clean'] = baseline
    initialize_reference(root, baseline, baseline_run)
    return root


def mock_learner_entry(monkeypatch, expected_exit=0):
    real_run = subprocess.run
    observed = []
    def run(command, **kwargs):
        if len(command) > 1 and command[1] == 'main.py':
            checkout = Path(kwargs['cwd'])
            commit = git(checkout, 'rev-parse', 'HEAD')
            assert not git(checkout, 'status', '--porcelain')
            assert (checkout / 'main.py').read_text() == subprocess.check_output(
                ['git', '-C', str(checkout), 'show', f'{commit}:main.py'], text=True)
            assert kwargs['env']['MLFLOW_TRACKING_URI'].startswith('sqlite:///')
            assert kwargs['env']['MLFLOW_DISABLE_TELEMETRY'] == 'true'
            pipeline_run = create_run(checkout, {'sample': command[2].split('=', 1)[1]})
            with stage(checkout, pipeline_run, 'generic_entry') as record:
                output = pipeline_run / 'revision.txt'
                output.write_text(commit)
                record['outputs']['revision'] = output
            observed.append((commit, pipeline_run.name))
            return subprocess.CompletedProcess(command, expected_exit)
        return real_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    return observed


def test_two_versions_preserve_dirty_checkout_and_relocate_bundle(tmp_path, monkeypatch):
    root = fixture_repo(tmp_path)
    original_head = git(root, 'rev-parse', 'HEAD')
    dirty_file = root / 'README.md'
    dirty_file.write_text('user work must survive\n')
    observed = mock_learner_entry(monkeypatch)
    assert run_version(root, '1.0.0', 'sample2.csv') == 0
    assert run_version(root, '1.0.1', 'sample2.csv') == 0
    assert observed[0][0] != observed[1][0]
    assert git(root, 'rev-parse', 'HEAD') == original_head
    assert dirty_file.read_text() == 'user work must survive\n'
    versions = read_json(root / 'artifacts/versions/versions.json')
    assert set(versions['versions']) == {'1.0.0', '1.0.1'}
    for version in ['1.0.0', '1.0.1']:
        entry = versions['versions'][version]
        evidence = read_json(root / entry['executions'][0]['path'])
        assert evidence['commit'] == git(root, 'rev-parse', f'{version}^{{commit}}')
        assert evidence['exit_status'] == 0
        manifest = read_json(root / evidence['manifests'][0]['path'])
        assert manifest['code'] == {'commit': evidence['commit'], 'dirty': False}
    relocated = tmp_path / 'relocated'
    shutil.copytree(root, relocated, ignore=shutil.ignore_patterns('.git'))
    assert run_version(relocated, '1.0.0', 'sample2.csv') == 0
    assert observed[-1][0] == observed[0][0]


def test_failed_entry_preserves_nonzero_evidence(tmp_path, monkeypatch):
    root = fixture_repo(tmp_path)
    mock_learner_entry(monkeypatch, expected_exit=3)
    assert run_version(root, '1.0.0', 'sample2.csv') == 3
    versions = read_json(root / 'artifacts/versions/versions.json')
    evidence = read_json(root / versions['versions']['1.0.0']['executions'][0]['path'])
    assert evidence['exit_status'] == 3
    audit = root / 'artifacts/runs' / evidence['execution_run_id']
    assert read_manifest(root, audit)['status'] == 'failed'
