"""Small explicit IO helpers, with no project-root or working-directory assumptions."""

import json
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_config(path):
    return json.loads(Path(path).read_text())


def array_stats(values):
    x = np.asarray(values)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }
