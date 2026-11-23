import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_obsolete_top_level_numerics_package():
    assert not (ROOT / "numerics").exists()


def test_common_package_does_not_import_examples():
    for path in (ROOT / "src/contgreedy").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in {"examples", "experiments"}, (
                    path
                )


def test_all_sources_parse():
    for directory in ["src", "examples", "fae", "tools"]:
        for path in (ROOT / directory).rglob("*.py"):
            compile(path.read_text(), str(path), "exec")
