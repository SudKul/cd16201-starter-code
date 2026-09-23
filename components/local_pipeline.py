"""Small local-file interface shared by pipeline components."""
import hashlib
import json
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def resolve_path(project_root, path, must_exist=False):
    root = Path(project_root).resolve()
    candidate = Path(path)
    candidate = (root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path must be inside project root: {path}")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Required input is missing: {candidate}")
    return candidate


def relative_path(project_root, path):
    return resolve_path(project_root, path).relative_to(Path(project_root).resolve()).as_posix()


def checksum(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob('*')):
            if child.is_symlink():
                raise ValueError(f"Artifact contains a symbolic link: {child}")
            if child.is_file():
                digest.update(child.relative_to(path).as_posix().encode() + b'\0')
                digest.update(bytes.fromhex(checksum(child)))
    else:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def artifact_record(project_root, path):
    resolved = resolve_path(project_root, path, must_exist=True)
    return {'path': relative_path(project_root, resolved), 'sha256': checksum(resolved)}


def load_artifact(project_root, record):
    path = resolve_path(project_root, record['path'], must_exist=True)
    if checksum(path) != record['sha256']:
        raise ValueError(f"Artifact checksum mismatch: {record['path']}")
    return path


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    """Create a JSON artifact exclusively; never replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _save_manifest(path, value):
    temporary = path.with_name(f'.manifest-{uuid.uuid4().hex}.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def create_run(project_root, config, run_id=None):
    root = Path(project_root).resolve()
    run_id = run_id or uuid.uuid4().hex
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_id must be a single directory name')
    run_dir = resolve_path(root, Path('artifacts/runs') / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain'], text=True))
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    write_json(run_dir / 'manifest.json', {
        'schema_version': 1, 'run_id': run_id, 'status': 'running',
        'config': config, 'code': {'commit': commit, 'dirty': dirty},
        'created_at': datetime.now(timezone.utc).isoformat(), 'stages': {},
    })
    return run_dir


def read_manifest(project_root, run_dir):
    directory = resolve_path(project_root, run_dir, must_exist=True)
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('schema_version') != 1 or manifest.get('run_id') != directory.name:
        raise ValueError(f"Invalid run manifest: {directory}")
    return manifest


def _records(project_root, paths):
    records = {}
    for name, path in (paths or {}).items():
        if isinstance(path, dict):
            records[name] = artifact_record(project_root, load_artifact(project_root, path))
        else:
            records[name] = artifact_record(project_root, path)
    return records


def update_stage(project_root, run_dir, stage, status, inputs=None, outputs=None, details=None):
    directory = resolve_path(project_root, run_dir, must_exist=True)
    manifest = read_manifest(project_root, directory)
    previous = manifest['stages'].get(stage)
    if status not in {'running', 'completed', 'failed'}:
        raise ValueError(f"Invalid stage status: {status}")
    if previous and previous['status'] in {'completed', 'failed'}:
        raise ValueError(f"Stage {stage} already {previous['status']}; create a new run")
    if status == 'running' and previous:
        raise ValueError(f"Stage {stage} already started; create a new run")
    output_records = _records(project_root, outputs)
    for record in output_records.values():
        if not resolve_path(project_root, record['path']).is_relative_to(directory):
            raise ValueError('Stage outputs must belong to their run directory')
    manifest['stages'][stage] = {
        'status': status, 'inputs': _records(project_root, inputs),
        'outputs': output_records, 'details': details or {},
    }
    manifest['status'] = ('failed' if any(s['status'] == 'failed' for s in manifest['stages'].values()) else
                          'completed' if all(s['status'] == 'completed' for s in manifest['stages'].values())
                          else 'running')
    _save_manifest(directory / 'manifest.json', manifest)


@contextmanager
def stage(project_root, run_dir, name, inputs=None):
    """Record stage outcome; callers populate outputs (paths) and details."""
    record = {'outputs': {}, 'details': {}}
    update_stage(project_root, run_dir, name, 'running')
    try:
        # Validate after starting so missing inputs still leave failed-run evidence.
        input_records = _records(project_root, inputs)
        yield record
        update_stage(project_root, run_dir, name, 'completed', inputs=input_records,
                     outputs=record['outputs'], details=record['details'])
    except BaseException as error:
        update_stage(project_root, run_dir, name, 'failed',
                     details={'error': f'{type(error).__name__}: {error}'})
        raise


def require_output(project_root, run_dir, stage_name, output_name):
    manifest = read_manifest(project_root, run_dir)
    entry = manifest['stages'].get(stage_name, {})
    if entry.get('status') != 'completed':
        raise ValueError(f"Required stage {stage_name} is not completed in {run_dir}")
    record = entry.get('outputs', {}).get(output_name)
    if not record:
        raise ValueError(f"Missing {output_name} output from {stage_name}")
    return load_artifact(project_root, record)


def run_output(project_root, run_dir, output):
    directory = resolve_path(project_root, run_dir, must_exist=True)
    path = resolve_path(project_root, output)
    if not path.is_relative_to(directory) or path == directory:
        raise ValueError('Output must be inside its run directory')
    if path.exists():
        raise FileExistsError(f"Output already exists; create a new run: {path}")
    return path
