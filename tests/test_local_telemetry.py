"""MLflow entry imports must stay offline even without a prepared shell."""
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    'main.py',
    'src/train_random_forest/run.py',
    'components/test_regression_model/run.py',
)


@pytest.mark.parametrize('entrypoint', ENTRYPOINTS)
def test_direct_entry_import_does_not_open_internet_socket(entrypoint, tmp_path):
    environment = os.environ.copy()
    for name in (
        'PYTEST_CURRENT_TEST', 'CI', 'GITHUB_ACTIONS', 'CIRCLECI', 'GITLAB_CI',
        'JENKINS_URL', 'TRAVIS', 'TF_BUILD', 'BITBUCKET_BUILD_NUMBER',
        'CODEBUILD_BUILD_ARN', 'BUILDKITE', 'RUNBOT_HOST_URL', 'RUNBOT_BUILD_NAME',
        'RUNBOT_WORKER_ID', 'DO_NOT_TRACK', '_MLFLOW_TESTING_TELEMETRY',
    ):
        environment.pop(name, None)
    environment['MLFLOW_DISABLE_TELEMETRY'] = 'false'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
    script = '''
import os
import runpy
import socket
import sys
import time

attempted = []
real_connect = socket.socket.connect
real_getaddrinfo = socket.getaddrinfo

def blocked_connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        attempted.append(('connect', str(address)))
        raise OSError('internet socket blocked by test')
    return real_connect(sock, address)

def blocked_getaddrinfo(*args, **kwargs):
    attempted.append(('dns', str(args[0])))
    raise OSError('DNS blocked by test')

socket.socket.connect = blocked_connect
socket.getaddrinfo = blocked_getaddrinfo
runpy.run_path(sys.argv[1], run_name='telemetry_import_probe')
from mlflow.telemetry import get_telemetry_client
time.sleep(0.2)
assert get_telemetry_client() is None, 'MLflow telemetry client started'
assert not attempted, f'internet access attempted: {attempted}'
assert os.environ['MLFLOW_DISABLE_TELEMETRY'] == 'true'
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(ROOT / entrypoint)],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
