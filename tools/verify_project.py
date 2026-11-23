"""Static checks of maintained source paths; no numerical experiments are run."""

from pathlib import Path
import ast, importlib.util, json, re

ROOT = Path(__file__).resolve().parents[1]


def read_tex_with_inputs(path, tex_root=None, active=()):
    """Expand local TeX inputs before checking cross-file labels and references."""
    path = path.resolve()
    tex_root = path.parent if tex_root is None else tex_root
    if path in active:
        raise AssertionError(f"Cyclic TeX input: {path}")

    def expand(match):
        child = tex_root / match.group(1)
        if not child.suffix:
            child = child.with_suffix(".tex")
        return read_tex_with_inputs(child, tex_root, (*active, path))

    return re.sub(
        r"\\(?:input|include)\{([^}]+)\}", expand, path.read_text()
    )


def main():
    roots = [
        ROOT / x
        for x in ("src", "examples", "tools", "experiments", "tests")
    ]
    files = [p for root in roots for p in root.rglob("*.py")] + [ROOT / "run.py"]
    missing = []
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                base = node.module.split(".")[0]
                if base not in {
                    "contgreedy",
                    "examples",
                    "tools",
                    "experiments",
                }:
                    continue
                local = (ROOT / "src" if base == "contgreedy" else ROOT).joinpath(
                    *node.module.split(".")
                )
                if (
                    not local.with_suffix(".py").exists()
                    and not (local / "__init__.py").exists()
                ):
                    missing.append((str(path), node.module))
                if path.is_relative_to(ROOT / "src") and base in {
                    "examples",
                    "experiments",
                }:
                    raise AssertionError(f"Common code imports an example: {path}")
    assert not missing, missing
    tex = ROOT / "paper/continuous_greedy.tex"
    text = read_tex_with_inputs(tex)
    assets = re.findall(
        r"\\(?:includegraphics|input|include)(?:\[[^\]]*\])?\{([^}]+)\}", text
    )
    for name in assets:
        base = tex.parent / name
        assert any(
            Path(str(base) + s).exists()
            for s in ("", ".tex", ".png", ".pdf", ".jpg", ".eps")
        ), name
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text)) <= set(labels)
    commands = json.loads((ROOT / "configs/commands.json").read_text())
    for item in commands.values():
        assert ROOT.joinpath(*item["module"].split(".")).with_suffix(".py").exists()
        assert (ROOT / item["working_directory"]).is_dir()
    print(
        json.dumps(
            {
                "python_files": len(files),
                "local_imports": "resolved",
                "common_dependency_direction": "passed",
                "tex_external_assets": assets,
                "tex_cross_references": "resolved",
                "experiment_commands": len(commands),
                "installed_dependencies": {
                    n: bool(importlib.util.find_spec(n))
                    for n in (
                        "numpy",
                        "scipy",
                        "jax",
                        "matplotlib",
                        "threadpoolctl",
                        "skfem",
                        "torch",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
