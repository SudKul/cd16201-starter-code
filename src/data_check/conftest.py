"""Load current and fixed-reference rental data from explicit local paths."""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from components.init_reference import load_reference
from components.local_pipeline import resolve_path


def pytest_addoption(parser):
    parser.addoption('--project_root', default=str(Path(__file__).resolve().parents[2]))
    parser.addoption('--csv')
    parser.addoption('--ref')
    parser.addoption('--kl_threshold')
    parser.addoption('--min_price')
    parser.addoption('--max_price')


def pytest_configure(config):
    root = Path(config.getoption('project_root')).resolve()
    values = {}
    for name in ('csv', 'ref'):
        raw = config.getoption(name)
        if not raw:
            raise pytest.UsageError(f'--{name} is required')
        try:
            path = resolve_path(root, raw, must_exist=True)
        except (ValueError, FileNotFoundError) as error:
            raise pytest.UsageError(str(error)) from error
        if not path.is_file():
            raise pytest.UsageError(f'--{name} must identify a CSV file: {path}')
        values[name] = path

    for name in ('kl_threshold', 'min_price', 'max_price'):
        raw = config.getoption(name)
        if raw is None:
            raise pytest.UsageError(f'--{name} is required')
        try:
            value = float(raw)
        except ValueError as error:
            raise pytest.UsageError(f'--{name} must be a finite number') from error
        if not math.isfinite(value):
            raise pytest.UsageError(f'--{name} must be a finite number')
        values[name] = value
    if values['kl_threshold'] < 0:
        raise pytest.UsageError('--kl_threshold must be nonnegative')
    if values['min_price'] >= values['max_price']:
        raise pytest.UsageError('--min_price must be lower than --max_price')
    if values['csv'] == values['ref']:
        raise pytest.UsageError('--csv must differ from the fixed --ref snapshot')
    try:
        fixed = load_reference(root)
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise pytest.UsageError(str(error)) from error
    if values['ref'] != fixed:
        raise pytest.UsageError('--ref must be the initialized fixed reference snapshot')
    config._local_data_options = values


@pytest.fixture(scope='session')
def data(request):
    return pd.read_csv(request.config._local_data_options['csv'])


@pytest.fixture(scope='session')
def ref_data(request):
    return pd.read_csv(request.config._local_data_options['ref'])


@pytest.fixture(scope='session')
def kl_threshold(request):
    return request.config._local_data_options['kl_threshold']


@pytest.fixture(scope='session')
def min_price(request):
    return request.config._local_data_options['min_price']


@pytest.fixture(scope='session')
def max_price(request):
    return request.config._local_data_options['max_price']
