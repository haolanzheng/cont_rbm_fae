"""Verify the portable bundle, check CUDA, and resume the unchanged trainer.

--verify-only checks files/checkpoint on CPU without starting training.
Normal execution requires CUDA and performs one batch forward/backward (no
optimizer update) before resuming from the last complete epoch.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from .model import KernelPoolFAE, parameter_counts, trapezoid_weights

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/fae128"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/fae128.json").read_text())
    manifest_path = ROOT / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["sha256"].items():
            # Run outputs legitimately change during subsequent resumes.
            if not name.startswith(("src/contgreedy/fae/", "configs/", "data/")):
                continue
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Bundle file changed: {name}")
    checkpoint = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"] != config:
        raise RuntimeError("Checkpoint and configuration differ")
    epoch = int(checkpoint["epoch"])
    expected_steps = (
        epoch * config["training"]["train_size"] // config["training"]["batch_size"]
    )
    optimizer_steps = {
        int(state["step"])
        for state in checkpoint["optimizer"]["state"].values()
        if "step" in state
    }
    if optimizer_steps != {expected_steps}:
        raise RuntimeError(f"Unexpected optimizer steps: {optimizer_steps}")
    data = ROOT / "data" / config["data_directory"]
    metadata = json.loads((data / "metadata.json").read_text())
    fields = np.load(data / "fields.npy", mmap_mode="r")
    if list(fields.shape) != metadata["shape"]:
        raise RuntimeError("Data shape does not match metadata")
    model = KernelPoolFAE(**config["model"])
    model.load_state_dict(checkpoint["model"])
    details = {
        "checkpoint_epoch": epoch,
        "completed_updates": expected_steps,
        "target_epochs": config["training"]["epochs"],
        "data_shape": list(fields.shape),
        "parameters": parameter_counts(model),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "verification": "passed",
    }
    print(json.dumps(details), flush=True)
    if args.verify_only:
        return
    if epoch >= config["training"]["epochs"]:
        raise RuntimeError(
            "All requested epochs are already complete; inspect metrics.json"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run on an allocated GPU with CUDA-enabled PyTorch."
        )
    torch.set_num_threads(config["training"]["threads"])
    print(
        json.dumps(
            {"gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda}
        ),
        flush=True,
    )
    model = model.cuda()
    n = metadata["grid_resolution"]
    batch = config["training"]["batch_size"]
    raw = torch.from_numpy(np.array(fields[:batch], copy=True)).cuda().reshape(-1, n, n)
    target = (raw - metadata["mean"]) / metadata["std"]
    pred, code = model(target)
    weights = trapezoid_weights(n, n, device=target.device)
    loss = 0.5 * ((pred - target).flatten(1).square() * weights).sum(1).mean()
    loss = loss + config["training"]["latent_penalty"] * code.square().sum(1).mean()
    loss.backward()
    if not torch.isfinite(loss) or any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        raise RuntimeError("Non-finite CUDA forward/backward check")
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "cuda_forward_backward": "passed",
                "batch_size": batch,
                "optimizer_updates_during_check": 0,
            }
        ),
        flush=True,
    )
    del model, checkpoint, raw, target, pred, code, loss, weights
    torch.cuda.empty_cache()
    # Trainer resets/restores its RNG and optimizer; this check cannot change
    # its initial state. CUDA and CPU are not promised bitwise equivalent.
    sys.argv = ["fae.training", "--resume"]
    runpy.run_module("fae.training", run_name="__main__")


if __name__ == "__main__":
    # Prevent concurrent GPU jobs from overwriting this one run directory.
    with (RUN / "gpu_train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                "Another GPU launcher is already using this run directory"
            ) from exc
        main()
