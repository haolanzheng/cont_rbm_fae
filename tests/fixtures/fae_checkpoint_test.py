"""Checkpoint migration: original model is a frozen test-only oracle."""

import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from contgreedy.fae.model import KernelPoolFAE, parameter_counts, trapezoid_weights
from contgreedy.fae.representation import FAERepresentation

ROOT = Path(__file__).resolve().parents[1]


def reference_model():
    spec = importlib.util.spec_from_file_location(
        "fae_reference", Path(__file__).parent / "fixtures/fae_reference_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KernelPoolFAE


def test_checkpoint_prediction_gradient_and_adam_step():
    torch.set_num_threads(2)
    state = torch.load(
        ROOT / "runs/fae128/last.pt", map_location="cpu", weights_only=False
    )
    new = KernelPoolFAE(**state["config"]["model"])
    old = reference_model()(**state["config"]["model"])
    assert list(new.state_dict()) == list(old.state_dict())
    assert [n for n, _ in new.named_parameters()] == [
        n for n, _ in old.named_parameters()
    ]
    assert parameter_counts(new)["real_scalars"] == 496463
    fields = np.load(ROOT / "data/data_80_formal1024/fields.npy", mmap_mode="r")
    import json

    meta = json.loads((ROOT / "data/data_80_formal1024/metadata.json").read_text())
    raw = torch.tensor(np.array(fields[:2], copy=True)).reshape(2, 80, 80)
    target = (raw - meta["mean"]) / meta["std"]
    results = []
    for model in [old, new]:
        model.load_state_dict(state["model"])
        opt = torch.optim.Adam(model.parameters())
        opt.load_state_dict(copy.deepcopy(state["optimizer"]))
        prediction, z = model(target)
        loss = (
            0.5
            * ((prediction - target).flatten(1).square() * trapezoid_weights(80, 80))
            .sum(1)
            .mean()
            + 1e-4 * z.square().sum(1).mean()
        )
        opt.zero_grad()
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        results.append(
            (
                prediction.detach(),
                z.detach(),
                gradients,
                [p.detach().clone() for p in model.parameters()],
            )
        )
    for a, b in zip(results[0], results[1]):
        if isinstance(a, list):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_latent_pullback():
    torch.manual_seed(9)
    model = KernelPoolFAE().double()
    representation = FAERepresentation(model, mean=0.3, std=1.7)
    z = torch.randn(1, 128, dtype=torch.float64)
    direction = torch.randn_like(z)
    g = torch.randn(1, 5, 5, dtype=torch.float64)
    pullback = representation.pullback(z, g, (5, 5))
    epsilon = 1e-5
    fd = (
        (
            representation.decode(z + epsilon * direction, (5, 5))
            - representation.decode(z - epsilon * direction, (5, 5))
        )
        * g
    ).sum() / (2 * epsilon)
    torch.testing.assert_close((pullback * direction).sum(), fd, rtol=1e-5, atol=1e-7)
    assert all(not p.requires_grad for p in model.parameters())
