"""Create one portable project archive without environments, caches or builds."""

from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    "__pycache__",
    "build",
    "qa",
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".so",
    ".o",
    ".aux",
    ".log",
    ".out",
    ".fls",
    ".fdb_latexmk",
    ".lock",
    ".tmp",
    ".zip",
}


def main():
    files = sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not EXCLUDED_DIRS.intersection(p.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)
        and p.suffix not in EXCLUDED_SUFFIXES
        and p.name != "MANIFEST.json"
    )
    manifest = {
        "project": ROOT.name,
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    files.append(ROOT / "MANIFEST.json")
    destination = ROOT.parent / (ROOT.name + ".zip")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for p in files:
            archive.write(p, str(Path(ROOT.name) / p.relative_to(ROOT)))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "archive": str(destination),
                "bytes": destination.stat().st_size,
                "files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
