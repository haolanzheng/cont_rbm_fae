"""Epoch-based, restartable training of the compact kernel-pooling FAE.

--stop-after-epochs pauses a run without shortening its LR schedule or pretending
that it completed 200 epochs. Test split is evaluated only after full training.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import KernelPoolFAE, parameter_counts, trapezoid_weights


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def atomic_checkpoint(path, state):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def stats(x):
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


@torch.no_grad()
def evaluate(model, fields, ids, mean, std, n, batch_size, device, weights):
    model.eval()
    errors, latent_norms = [], []
    for first in range(0, len(ids), batch_size):
        raw = torch.from_numpy(
            np.array(fields[ids[first : first + batch_size]], copy=True)
        ).to(device)
        inputs = (raw.reshape(-1, n, n) - mean) / std
        predicted, z = model(inputs)
        output = predicted.flatten(1) * std + mean
        er = (
            ((output - raw).square() * weights).sum(1)
            / (raw.square() * weights).sum(1).clamp_min(1e-12)
        ).sqrt()
        errors.extend(er.cpu().tolist())
        latent_norms.extend(z.square().sum(1).cpu().tolist())
    return {
        "n": len(errors),
        "relative_L2": stats(errors),
        "latent_squared_norm_mean": float(np.mean(latent_norms)),
    }


def train(
    cfg,
    *,
    data_directory,
    run_directory,
    resume=False,
    stop_after_epochs=0,
    device=None,
):
    """Train or resume using explicit paths. Configuration owns the full schedule."""
    t = cfg["training"]
    if not 1 <= t["train_size"] <= 8192 or t["batch_size"] not in (16, 32):
        raise ValueError("User limits: train_size<=8192 and batch_size 16 or 32")
    if t["train_size"] % t["batch_size"]:
        raise ValueError(
            "This run requires complete batches, avoiding an accidental batch of 1"
        )
    run = Path(run_directory)
    run.mkdir(parents=True, exist_ok=True)
    last = run / "last.pt"
    if last.exists() and not resume:
        raise ValueError("Run already has a checkpoint; use --resume or a new --tag")
    if resume and not last.exists():
        raise ValueError("No checkpoint available to resume")
    torch.set_num_threads(t["threads"])
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = Path(data_directory)
    metadata = json.loads((data / "metadata.json").read_text())
    fields = np.load(data / "fields.npy", mmap_mode="r")
    n = metadata["grid_resolution"]
    train0, train1 = metadata["splits"]["train"]
    if t["train_size"] > train1 - train0:
        raise ValueError("Insufficient training samples")
    train_ids = np.arange(train0, train0 + t["train_size"])
    val_ids = np.arange(*metadata["splits"]["validation"])
    test_ids = np.arange(*metadata["splits"]["test"])
    mean, std = metadata["mean"], metadata["std"]
    weights = trapezoid_weights(n, n, device=device)
    model = KernelPoolFAE(**cfg["model"]).to(device)
    counts = parameter_counts(model)
    if counts["real_scalars"] > t["max_real_parameters"]:
        raise ValueError("Model exceeds real-parameter budget")
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    start_epoch, best, history, previous_seconds = 0, float("inf"), [], 0.0
    if resume:
        state = torch.load(last, map_location=device, weights_only=False)
        if state["config"] != cfg:
            raise ValueError(
                "Configuration changed; resume requires exact configuration"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best = state["epoch"], state["best"]
        history, previous_seconds = state["history"], state["elapsed_seconds"]
    steps_per_epoch = len(train_ids) // t["batch_size"]
    total_steps = steps_per_epoch * t["epochs"]
    warmup = steps_per_epoch * t["warmup_epochs"]
    final_epoch = min(stop_after_epochs or t["epochs"], t["epochs"])
    if final_epoch <= start_epoch:
        raise ValueError("Requested stopping epoch is already complete")
    source_root = Path(__file__).resolve().parent
    provenance = {
        f"contgreedy.fae.{name[:-3]}": hashlib.sha256(
            (source_root / name).read_bytes()
        ).hexdigest()
        for name in ("model.py", "training.py")
    }
    atomic_json(run / "config.json", cfg)
    atomic_json(
        run / "provenance.json",
        {
            "sources_sha256": provenance,
            "parameters": counts,
            "objective": "0.5*quadrature_L2_squared + beta*sum(z^2)",
            "normalization": {"mean": mean, "std": std},
            "data_metadata": metadata,
        },
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": counts,
                "start_epoch": start_epoch,
                "stop_epoch": final_epoch,
                "planned_epochs": t["epochs"],
                "planned_updates": total_steps,
            }
        ),
        flush=True,
    )
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, final_epoch + 1):
        model.train()
        order = rng.permutation(train_ids)
        losses, recons, penalties = [], [], []
        for b in range(steps_per_epoch):
            step = (epoch - 1) * steps_per_epoch + b + 1
            if step <= warmup:
                lr = t["learning_rate"] * step / warmup
            else:
                frac = (step - warmup) / max(1, total_steps - warmup)
                lr = t["minimum_learning_rate"] + 0.5 * (
                    t["learning_rate"] - t["minimum_learning_rate"]
                ) * (1 + math.cos(math.pi * frac))
            for group in opt.param_groups:
                group["lr"] = lr
            ids = order[b * t["batch_size"] : (b + 1) * t["batch_size"]]
            raw = (
                torch.from_numpy(np.array(fields[ids], copy=True))
                .to(device)
                .reshape(-1, n, n)
            )
            target = (raw - mean) / std
            pred, z = model(target)
            recon = 0.5 * ((pred - target).flatten(1).square() * weights).sum(1).mean()
            penalty = z.square().sum(1).mean()
            loss = recon + t["latent_penalty"] * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {b}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), t["gradient_clip"], error_if_nonfinite=True
            )
            opt.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon.detach()))
            penalties.append(float(penalty.detach()))
        row = {
            "epoch": epoch,
            "step": epoch * steps_per_epoch,
            "loss": float(np.mean(losses)),
            "reconstruction": float(np.mean(recons)),
            "latent_squared_norm": float(np.mean(penalties)),
            "learning_rate": lr,
            "last_gradient_norm": float(grad),
            "elapsed_seconds": previous_seconds + time.perf_counter() - started,
        }
        if (
            epoch == 1
            or epoch % t["evaluation_interval_epochs"] == 0
            or epoch == final_epoch
        ):
            # Validation chooses the checkpoint; training samples never substitute for it.
            row["validation"] = evaluate(
                model, fields, val_ids, mean, std, n, t["batch_size"], device, weights
            )
            row["train_monitor_128"] = evaluate(
                model,
                fields,
                train_ids[:128],
                mean,
                std,
                n,
                t["batch_size"],
                device,
                weights,
            )
            score = row["validation"]["relative_L2"]["mean"]
            if score < best:
                best = score
                atomic_checkpoint(
                    run / "best.pt",
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "config": cfg,
                        "normalization": {"mean": mean, "std": std},
                        "validation": row["validation"],
                    },
                )
        history.append(row)
        atomic_checkpoint(
            last,
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "config": cfg,
                "best": best,
                "history": history,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
                "elapsed_seconds": previous_seconds + time.perf_counter() - started,
            },
        )
        atomic_json(run / "history.json", history)
        atomic_json(
            run / "status.json",
            {
                "status": "running",
                "epoch": epoch,
                "planned_epochs": t["epochs"],
                "completed_updates": epoch * steps_per_epoch,
                "best_validation_relative_L2": best,
            },
        )
        print("EPOCH", json.dumps(row), flush=True)
    result = {
        "status": "completed" if final_epoch == t["epochs"] else "paused",
        "epoch": final_epoch,
        "planned_epochs": t["epochs"],
        "completed_updates": final_epoch * steps_per_epoch,
        "parameters": counts,
        "last_epoch": history[-1],
        "best_validation_relative_L2": best,
    }
    if result["status"] == "completed":
        state = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        result["selected_epoch"] = state["epoch"]
        result["test"] = evaluate(
            model, fields, test_ids, mean, std, n, t["batch_size"], device, weights
        )
        atomic_json(run / "metrics.json", result)
    atomic_json(run / "status.json", result)
    print("FINAL_STATUS", json.dumps(result), flush=True)

    return result
