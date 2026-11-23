#!/usr/bin/env python3
"""One project entry point; no training or numerical experiment starts on import."""

from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def build_bgk():
    source = ROOT / "examples/bgk/bgk_exact_kernel.cpp"
    target = source.with_suffix(".so")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    compiler = os.environ.get("CXX", "g++")
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(target),
        ],
        check=True,
    )


def main():
    commands = json.loads((ROOT / "configs/commands.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="list")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "list":
        print("paper          Compile the self-contained manuscript")
        print("fae-check      Run pooling, spectral-transfer and gradient checks")
        print("fae-train      Fresh JAX/GPU training; requires --run-dir")
        print("fae-data       Generate inputs only if the requested data are absent")
        print("verify         Check local import closure and TeX asset dependencies")
        print("package        Build a clean portable ZIP and its SHA-256 manifest")
        print("build-bgk      Compile the platform-specific BGK kernel")
        if "--all" in args.arguments:
            for name, value in sorted(commands.items()):
                print(f"{name:<38} {value['module']}")
        else:
            print(
                "Use 'python run.py list --all' for numerical experiments and report utilities."
            )
        return
    if args.command == "paper":
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-outdir=build",
                "continuous_greedy.tex",
            ],
            cwd=ROOT / "paper",
            check=True,
        )
        shutil.copy2(
            ROOT / "paper/build/continuous_greedy.pdf",
            ROOT / "paper/continuous_greedy.pdf",
        )
        return
    if args.command == "build-bgk":
        build_bgk()
        return
    specials = {
        "fae-check": ("examples.hyperelasticity.check_fae", []),
        "fae-train": ("examples.hyperelasticity.train_fae", []),
        "fae-data": ("examples.hyperelasticity.data", []),
        "verify": ("tools.verify_project", []),
        "package": ("tools.package", []),
    }
    if args.command in specials:
        module, options = specials[args.command]
        cwd = ROOT
    elif args.command in commands:
        task = commands[args.command]
        module, options = task["module"], []
        cwd = ROOT / task["working_directory"]
        if module.startswith("examples.bgk."):
            build_bgk()
    else:
        parser.error(f"Unknown command {args.command!r}; use list --all")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        + os.pathsep
        + str(ROOT)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(
        [sys.executable, "-m", module, *options, *args.arguments], cwd=cwd, env=env
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
