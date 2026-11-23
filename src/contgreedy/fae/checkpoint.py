"""Required common utilities; Flax-native artifacts avoid a dill dependency."""
import json
from pathlib import Path
import jax
from flax import serialization


def get_n_params(variables):
    return sum(x.size for x in jax.tree_util.tree_leaves(variables))


def write_json(path, value):
    """Atomically replace a JSON artifact."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def save_model_results(results, model_dir, metadata=None):
    """Save the selected state; construct the model from the adjacent config.json.

    The original function pickled Python model/optimizer closures with dill.
    Only its artifact hook is adapted here; numerical training is unchanged.
    """
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "model.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(results["state"]))
    temporary.replace(directory / "model.msgpack")
    summary = {
        "metadata": metadata,
        "best_epoch": results["best_epoch"],
        "best_metric_name": results["best_metric_name"],
        "best_metric": results["best_metric"],
        "training_loss_history": [float(x) for x in results["training_loss_history"]],
        "metrics_history": {k: [float(x) for x in v] for k,v in results["metrics_history"].items()},
        "restore": "build from config.json, then flax.serialization.from_bytes(template_state, model.msgpack)",
    }
    write_json(directory / "model_results.json", summary)
