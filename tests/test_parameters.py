"""Executed constants are checked against preserved protocols and TeX anchors."""

import json
from pathlib import Path
import numpy as np
from examples.thermal_fin.hf import Fin
from contgreedy.data.grf import matrices

ROOT = Path(__file__).resolve().parents[1]


def test_spectral_presets_match_original_records():
    for path in (ROOT / "examples/baseline/configs").glob("paper_*.json"):
        c = json.loads(path.read_text())
        saved = json.loads((ROOT / c["source"]).read_text())["config"]
        assert c["problem"] == dict(
            grid=saved["grid"],
            dimension=saved["d"],
            contrast=saved["contrast"],
            spectrum_ratio=saved["spectrum_ratio"],
            load_mode=saved["load_mode"],
        )
        assert c["rank"] == saved["rank"]
        assert c["seeds"] == list(range(1000, 1000 + saved["seeds"]))
        assert c["direct_p_audit_samples"] == saved["audit_samples"]
        assert c["q_spectral_starts"] == 30000
        assert c["adam_policy"]["return_best"] is True


def test_fin_physical_parameters_and_dimensions():
    for dimension, q in [(11, 20), (21, 35), (33, 49)]:
        p = Fin(dimension, 4)
        assert p.q == q
        assert p.f.sum() == 1.0
        np.testing.assert_allclose(p.physical(-np.ones(dimension))[0], p.lo)
        np.testing.assert_allclose(p.physical(np.ones(dimension))[0], p.hi)
        assert p.lo[p.m] == 0.01 and p.hi[p.m] == 5
    p = Fin(11, 32)
    assert p.n == 10017 and len(p.rawparts) == 49


def test_fae_configuration_matches_tex():
    config = json.loads((ROOT / "configs/fae128.json").read_text())
    assert config == json.loads(
        (ROOT / "examples/hyperelasticity/configs/fae128.json").read_text()
    )
    model = config["model"]
    training = config["training"]
    assert (
        model["latent_dimension"] == 128
        and model["width"] == 16
        and model["modes"] == 8
    )
    assert model["decoder_width"] == 96 and model["decoder_blocks"] == 3
    assert model["pool_channels"] == 8 and model["kernel_hidden"] == 64
    assert training["train_size"] == 1024 and training["batch_size"] == 32
    assert training["epochs"] == 200 and training["warmup_epochs"] == 5
    assert (
        training["learning_rate"] == 5e-4 and training["minimum_learning_rate"] == 1e-5
    )
    assert training["latent_penalty"] == 1e-4 and training["gradient_clip"] == 1
    field = json.loads((ROOT / "configs/input_field.json").read_text())
    assert field["grf"]["delta"] == 0.4 and field["grf"]["gamma"] == 0.04
    assert field["data"] == dict(train=1024, validation=256, test=256)
    tex = (ROOT / "paper/continuous_greedy.tex").read_text()
    for label in ["tab:fae-config", "eq:fae-loss", "eq:elliptic-pde"]:
        assert label in tex


def test_grf_mass_and_robin_assembly():
    xy, tri, M, K, R, L = matrices(4)
    ones = np.ones(len(xy))
    np.testing.assert_allclose(ones @ M @ ones, 1)
    np.testing.assert_allclose(K @ ones, 0, atol=1e-14)
    np.testing.assert_allclose(ones @ R @ ones, 4)
