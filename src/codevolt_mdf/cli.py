from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ContractError, Experiment, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(prog="codevolt-mdf")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate an experiment manifest")
    validate.add_argument("manifest", type=Path)
    run = sub.add_parser("run", help="run a bounded experiment")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()

    try:
        if args.command == "validate":
            experiment = Experiment.from_dict(json.loads(args.manifest.read_text()))
            experiment.validate()
            print(f"valid: {experiment.name}")
        else:
            run_dir = run_experiment(args.manifest, args.output)
            print(run_dir)
        return 0
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2
