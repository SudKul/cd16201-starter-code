#!/usr/bin/env python
"""Clean a local rental CSV within one pipeline run."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from components.local_pipeline import resolve_path, run_output, stage


def clean_price_and_dates(data, min_price, max_price):
    raise NotImplementedError(
        'TODO: filter prices to the configured range and clean date fields '
        'without changing the required input schema.'
    )


def clean_geographic_boundaries(data):
    # TODO for the later sample2 version: remove observations outside NYC bounds.
    return data


def go(args):
    with stage(args.project_root, args.run_dir, 'basic_cleaning', inputs={'raw': args.input}) as record:
        if args.min_price >= args.max_price:
            raise ValueError('--min_price must be lower than --max_price')
        source = resolve_path(args.project_root, args.input, must_exist=True)
        output = run_output(args.project_root, args.run_dir, args.output)
        data = pd.read_csv(source)
        cleaned = clean_price_and_dates(data, args.min_price, args.max_price)
        cleaned = clean_geographic_boundaries(cleaned)
        if not isinstance(cleaned, pd.DataFrame):
            raise TypeError('Cleaning functions must return a pandas DataFrame')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            cleaned.to_csv(stream, index=False)
        record['outputs']['clean'] = output
        record['details']['min_price'] = args.min_price
        record['details']['max_price'] = args.max_price


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--run_dir', required=True, help='New run directory with manifest.json')
    parser.add_argument('--input', required=True, help='Input CSV inside the project root')
    parser.add_argument('--output', required=True, help='New clean CSV inside the run directory')
    parser.add_argument('--min_price', type=float, required=True)
    parser.add_argument('--max_price', type=float, required=True)
    go(parser.parse_args())
