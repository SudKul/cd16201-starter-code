"""Forward root MLproject parameters to Hydra without shell word splitting."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', required=True)
    parser.add_argument('--hydra_options', default='')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    overrides = shlex.split(args.hydra_options)
    command = [sys.executable, str(root / 'main.py'), f"main.steps='{args.steps}'", *overrides]
    return subprocess.call(command, cwd=root)


if __name__ == '__main__':
    raise SystemExit(main())
