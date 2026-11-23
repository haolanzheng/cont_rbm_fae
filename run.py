#!/usr/bin/env python3
"""Run the retained examples from the repository root."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    commands = json.loads((ROOT / "configs/commands.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="list")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "list":
        for name, task in commands.items():
            print(f"{name:<18} {task['description']}")
        return
    if args.command not in commands:
        parser.error(f"Unknown command {args.command!r}; use list")
    task = commands[args.command]
    entry = ["-m", task["module"]] if "module" in task else [str(ROOT / task["script"])]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run([sys.executable, *entry, *args.arguments], cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
