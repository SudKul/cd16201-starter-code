#!/usr/bin/env python
"""Copy a bundled rental dataset to a new, persistent run output."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from components.local_pipeline import artifact_record, resolve_path, run_output, stage


def go(args):
    with stage(args.project_root, args.run_dir, 'download') as record:
        if args.sample not in {'sample1.csv', 'sample2.csv'}:
            raise FileNotFoundError(f'Unknown bundled sample: {args.sample}; use sample1.csv or sample2.csv')
        source = resolve_path(args.project_root, f'components/get_data/data/{args.sample}', must_exist=True)
        output = run_output(args.project_root, args.run_dir, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, output.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        record['outputs']['raw'] = output
        record['details']['sample'] = args.sample
        # Keep the bundled source identity alongside the copied input evidence.
        record['details']['source'] = artifact_record(args.project_root, source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--run_dir', required=True, help='New run directory with manifest.json')
    parser.add_argument('--sample', required=True, help='Bundled sample1.csv or sample2.csv')
    parser.add_argument('--output', required=True, help='New CSV output inside run_dir')
    go(parser.parse_args())
