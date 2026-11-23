"""A temporary two-epoch run checks restart semantics without touching fae128."""

import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from contgreedy.fae.training import train


def test_trainer_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    fields = np.random.default_rng(81).normal(size=(36, 16 * 16)).astype("float32")
    np.save(data / "fields.npy", fields)
    meta = {
        "grid_resolution": 16,
        "shape": list(fields.shape),
        "mean": float(fields[:32].mean()),
        "std": float(fields[:32].std()),
        "splits": {"train": [0, 32], "validation": [32, 34], "test": [34, 36]},
    }
    (data / "metadata.json").write_text(json.dumps(meta))
    config = {
        "model": {
            "latent_dimension": 4,
            "width": 4,
            "modes": 2,
            "pool_channels": 2,
            "kernel_hidden": 8,
            "decoder_width": 12,
            "decoder_blocks": 1,
            "condition_width": 12,
            "coordinate_frequencies": 2,
        },
        "training": {
            "train_size": 32,
            "batch_size": 16,
            "epochs": 2,
            "warmup_epochs": 1,
            "learning_rate": 5e-4,
            "minimum_learning_rate": 1e-5,
            "latent_penalty": 1e-4,
            "gradient_clip": 1.0,
            "seed": 9,
            "threads": 2,
            "max_real_parameters": 2000000,
            "evaluation_interval_epochs": 1,
        },
    }
    train(config, data_directory=data, run_directory=tmp_path / "full", device="cpu")
    paused = train(
        config,
        data_directory=data,
        run_directory=tmp_path / "split",
        device="cpu",
        stop_after_epochs=1,
    )
    assert paused["status"] == "paused"
    assert not (tmp_path / "split/metrics.json").exists()
    completed = train(
        config,
        data_directory=data,
        run_directory=tmp_path / "split",
        device="cpu",
        resume=True,
    )
    assert completed["status"] == "completed" and completed["completed_updates"] == 4
    full = torch.load(tmp_path / "full/last.pt", map_location="cpu", weights_only=False)
    split = torch.load(
        tmp_path / "split/last.pt", map_location="cpu", weights_only=False
    )
    for key in full["model"]:
        torch.testing.assert_close(
            full["model"][key], split["model"][key], rtol=0, atol=0
        )
    assert completed["test"]["n"] == 2
