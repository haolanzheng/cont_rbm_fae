"""Run reconstruction-model checks for the hyperelastic input FAE."""

import json
from pathlib import Path

from contgreedy.fae.checks import run_checks


def main():
    config_path = Path(__file__).parent / "configs" / "fae128.json"
    config = json.loads(config_path.read_text())
    print(json.dumps(run_checks(config["model"]), indent=2))


if __name__ == "__main__":
    main()
