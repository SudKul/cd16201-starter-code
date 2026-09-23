#!/usr/bin/env python
"""Write reproducible local train/validation and held-out CSV datasets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from components.local_pipeline import (
    artifact_record, read_manifest, require_output, resolve_path,
    run_output, stage, write_json,
)


def go(args):
    with stage(args.project_root, args.run_dir, 'data_split', inputs={'source': args.input}) as record:
        if not 0 < args.test_size < 1:
            raise ValueError('test_size must be a fraction strictly between zero and one')
        source = resolve_path(args.project_root, args.input, must_exist=True)
        source_run = source.parent
        if require_output(args.project_root, source_run, 'basic_cleaning', 'clean') != source:
            raise ValueError('Split input must be the completed cleaning output of its source run')
        data = pd.read_csv(source)
        parameters = {'test_size': args.test_size, 'random_seed': args.random_seed,
                      'stratify_by': args.stratify_by}
        trainval, test = train_test_split(
            data, test_size=args.test_size, random_state=args.random_seed,
            stratify=data[args.stratify_by] if args.stratify_by != 'none' else None,
        )
        output = run_output(args.project_root, args.run_dir, args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        trainval.to_csv(output / 'trainval.csv', index=False, mode='x')
        test.to_csv(output / 'test.csv', index=False, mode='x')
        source_record = artifact_record(args.project_root, source)
        identity = {'source_sha256': source_record['sha256'], 'parameters': parameters}
        split_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        metadata = {
            'schema_version': 1, 'split_id': split_id,
            'source_run_id': read_manifest(args.project_root, source_run)['run_id'],
            'source': source_record, 'parameters': parameters,
            'trainval': artifact_record(args.project_root, output / 'trainval.csv'),
            'test': artifact_record(args.project_root, output / 'test.csv'),
        }
        write_json(output / 'split.json', metadata)
        record['outputs'].update(trainval=output / 'trainval.csv', test=output / 'test.csv',
                                 split_metadata=output / 'split.json')
        record['details'].update(parameters=parameters, split_id=split_id,
                                 source_run_id=metadata['source_run_id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--input', required=True, help='Completed clean.csv, relative to project root')
    parser.add_argument('--output_dir', required=True, help='New split directory inside run_dir')
    parser.add_argument('--test_size', type=float, required=True)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--stratify_by', default='none')
    go(parser.parse_args())
