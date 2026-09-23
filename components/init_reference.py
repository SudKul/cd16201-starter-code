"""Explicitly preserve one completed cleaning output as the fixed baseline."""
import argparse
import shutil
from pathlib import Path

from components.local_pipeline import (
    artifact_record, load_artifact, read_json, read_manifest, relative_path,
    require_output, resolve_path, write_json,
)


def initialize_reference(project_root, source, source_run, reference_dir='artifacts/reference'):
    source_path = resolve_path(project_root, source, must_exist=True)
    completed = require_output(project_root, source_run, 'basic_cleaning', 'clean')
    if source_path != completed:
        raise ValueError('Reference source must be the completed cleaning output of source_run')
    destination = resolve_path(project_root, reference_dir)
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / 'reference.csv'
    with source_path.open('rb') as reader, snapshot.open('xb') as writer:
        shutil.copyfileobj(reader, writer)
    metadata = {
        'schema_version': 1,
        'source_run_id': read_manifest(project_root, source_run)['run_id'],
        'source': artifact_record(project_root, source_path),
        'reference': artifact_record(project_root, snapshot),
    }
    if metadata['source']['sha256'] != metadata['reference']['sha256']:
        raise ValueError('Source changed during reference initialization; use a new reference directory')
    write_json(destination / 'reference.json', metadata)
    return metadata


def load_reference(project_root, reference_dir='artifacts/reference'):
    directory = resolve_path(project_root, reference_dir)
    if not (directory / 'reference.json').is_file():
        raise FileNotFoundError(
            'Fixed reference is missing. Initialize it explicitly with '
            'python -m components.init_reference --project_root ROOT '
            '--source RUN/clean.csv --source_run RUN')
    metadata = read_json(directory / 'reference.json')
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported reference metadata schema')
    expected = relative_path(project_root, directory / 'reference.csv')
    if metadata['reference']['path'] != expected:
        raise ValueError('Reference metadata points outside its snapshot directory')
    if metadata['reference']['sha256'] != metadata['source']['sha256']:
        raise ValueError('Reference checksum does not match its recorded source')
    return load_artifact(project_root, metadata['reference'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--source', required=True, help='Completed cleaned CSV, relative to project root')
    parser.add_argument('--source_run', required=True, help='Run that produced the cleaned CSV')
    parser.add_argument('--reference_dir', default='artifacts/reference')
    args = parser.parse_args()
    initialize_reference(args.project_root, args.source, args.source_run, args.reference_dir)


if __name__ == '__main__':
    main()
