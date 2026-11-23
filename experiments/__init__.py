"""Compatibility namespace; new code imports examples."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / "examples")]
