"""Replay an existing local version in isolation and retain portable evidence."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from components.init_reference import load_reference
from components.local_pipeline import (
    artifact_record, checksum, create_run, load_artifact, read_json, read_manifest, relative_path,
    resolve_path, update_stage, write_json,
)

VERSIONS = ('1.0.0', '1.0.1')


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def _replace_json(path, value):
    temporary = path.with_name(f'.{path.name}-{uuid.uuid4().hex}')
    write_json(temporary, value)
    temporary.replace(path)


def _copy_reference(root, checkout):
    """Copy only the baseline and its transitive recorded run prerequisites."""
    load_reference(root)
    pending = [Path('artifacts/reference/reference.json')]
    copied = set()
    while pending:
        relative = pending.pop()
        if relative.as_posix() in copied:
            continue
        source = resolve_path(root, relative, must_exist=True)
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if checksum(source) != checksum(target):
                raise ValueError(f'Reference dependency conflicts with tagged source: {relative}')
        elif source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=False)
        else:
            shutil.copy2(source, target)
        copied.add(relative.as_posix())
        if source.suffix != '.json':
            continue
        value = read_json(source)
        def visit(item):
            if isinstance(item, dict):
                if 'path' in item and 'sha256' in item:
                    artifact = load_artifact(root, item)
                    path = Path(relative_path(root, artifact))
                    pending.append(path)
                    if path.parts[:2] == ('artifacts', 'runs'):
                        pending.append(Path(*path.parts[:3]) / 'manifest.json')
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
        visit(value)
    return {path.name for path in (checkout / 'artifacts/runs').glob('*')} if (checkout / 'artifacts/runs').exists() else set()


def run_version(project_root, version, sample):
    root = Path(project_root).resolve()
    if version not in VERSIONS:
        raise ValueError('Version must be an existing local 1.0.0 or 1.0.1 tag')
    if sample not in {'sample1.csv', 'sample2.csv'}:
        raise ValueError('Sample must be sample1.csv or sample2.csv')
    load_reference(root)
    source = root if (root / '.git').exists() else resolve_path(root, 'artifacts/versions/source.bundle', must_exist=True)
    versions_dir = root / 'artifacts/versions'
    versions_dir.mkdir(parents=True, exist_ok=True)
    versions_file = versions_dir / 'versions.json'
    versions = read_json(versions_file) if versions_file.exists() else {'schema_version': 1, 'versions': {}}
    if source.is_file():
        if not versions.get('bundle'):
            raise ValueError('Portable source bundle requires versions.json checksum evidence')
        if load_artifact(root, versions['bundle']) != source:
            raise ValueError('Version metadata points to a different source bundle')
    with tempfile.TemporaryDirectory(prefix='cd16201-version-') as temporary:
        checkout = Path(temporary) / 'project'
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(source), str(checkout)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            commit = git(checkout, 'rev-parse', '--verify', f'refs/tags/{version}^{{commit}}')
        except subprocess.CalledProcessError as error:
            raise ValueError(f'Missing local tag {version}; create it after completing and committing your version') from error
        for recorded_tag, previous in versions['versions'].items():
            tagged_commit = git(checkout, 'rev-parse', '--verify', f'refs/tags/{recorded_tag}^{{commit}}')
            if previous['commit'] != tagged_commit:
                raise ValueError(f'Tag {recorded_tag} differs from recorded evidence; do not replace version markers')
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', commit],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if not (checkout / 'main.py').is_file():
            raise FileNotFoundError(f'Tag {version} has no main.py entry point')
        baseline_runs = _copy_reference(root, checkout)
        sample_path = resolve_path(checkout, f'components/get_data/data/{sample}', must_exist=True)
        config = resolve_path(checkout, 'config.yaml', must_exist=True)
        audit = create_run(root, {'version': version, 'sample': sample})
        shutil.copy2(config, audit / 'tagged-config.yaml')
        shutil.copy2(sample_path, audit / 'input.csv')
        command = ['python', 'main.py', f'etl.sample={sample}']
        update_stage(root, audit, 'version_run', 'running')
        environment = os.environ.copy()
        environment['MLFLOW_TRACKING_URI'] = 'sqlite:///' + str(checkout / 'artifacts/mlflow/mlflow.db')
        environment['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'
        (checkout / 'artifacts/mlflow').mkdir(parents=True, exist_ok=True)
        with (audit / 'stdout.log').open('x') as stdout, (audit / 'stderr.log').open('x') as stderr:
            completed = subprocess.run([sys.executable, *command[1:]], cwd=checkout, env=environment,
                                       stdout=stdout, stderr=stderr)
        manifests = []
        for directory in sorted((checkout / 'artifacts/runs').glob('*')):
            if directory.name in baseline_runs:
                continue
            if not directory.is_dir() or not (directory / 'manifest.json').is_file():
                continue
            manifest = read_manifest(checkout, directory)
            if manifest['code'] != {'commit': commit, 'dirty': False}:
                raise ValueError('Replayed manifest does not identify clean tagged source')
            destination = root / 'artifacts/runs' / directory.name
            if destination.exists():
                raise FileExistsError(f'Replayed run ID already exists: {directory.name}')
            shutil.copytree(directory, destination)
            manifests.append(artifact_record(root, destination / 'manifest.json'))
        evidence = {
            'schema_version': 1, 'version': version, 'commit': commit,
            'execution_run_id': audit.name, 'command': command,
            'python_version': sys.version.split()[0], 'exit_status': completed.returncode,
            'config': artifact_record(root, audit / 'tagged-config.yaml'),
            'input': artifact_record(root, audit / 'input.csv'),
            'sample': sample,
            'reference': read_json(checkout / 'artifacts/reference/reference.json')['reference'],
            'manifests': manifests,
        }
        write_json(audit / 'version.json', evidence)
        update_stage(root, audit, 'version_run',
                     'completed' if completed.returncode == 0 and manifests else 'failed',
                     outputs={key: audit / filename for key, filename in
                              [('evidence', 'version.json'), ('stdout', 'stdout.log'), ('stderr', 'stderr.log')]},
                     details={'version': version, 'commit': commit, 'exit_status': completed.returncode})
        available = [tag for tag in VERSIONS if tag in git(checkout, 'tag', '--list').splitlines()]
        temporary_bundle = versions_dir / f'.source-{uuid.uuid4().hex}.bundle'
        subprocess.run(['git', '-C', str(checkout), 'bundle', 'create', str(temporary_bundle),
                        *[f'refs/tags/{tag}' for tag in available]], check=True)
        temporary_bundle.replace(versions_dir / 'source.bundle')
        entry = versions['versions'].setdefault(version, {'commit': commit, 'executions': []})
        entry['executions'].append(artifact_record(root, audit / 'version.json'))
        versions['bundle'] = artifact_record(root, versions_dir / 'source.bundle')
        _replace_json(versions_file, versions)
        return completed.returncode if manifests else (completed.returncode or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--version', choices=VERSIONS, required=True)
    parser.add_argument('--sample', choices=['sample1.csv', 'sample2.csv'], default='sample2.csv')
    args = parser.parse_args()
    raise SystemExit(run_version(args.project_root, args.version, args.sample))


if __name__ == '__main__':
    main()
