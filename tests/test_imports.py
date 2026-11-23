"""Import every case module in an empty directory: no solver or report may run."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_no_cross_case_driver_imports():
    for path in (ROOT / "examples").rglob("*.py"):
        case = path.relative_to(ROOT / "examples").parts[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("examples.")
            ):
                assert node.module.split(".")[1] == case, (path, node.module)


def test_import_all_cases_from_empty_directory(tmp_path):
    modules = []
    for path in (ROOT / "examples").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if "hyperelasticity" in path.parts and not importlib.util.find_spec("torch"):
            continue
        if "flow_elasticity" in path.parts and not importlib.util.find_spec("skfem"):
            continue
        modules.append(".".join(path.relative_to(ROOT).with_suffix("").parts))
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({m!r})" for m in modules
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir()), (
        "Import produced files in the current directory"
    )
