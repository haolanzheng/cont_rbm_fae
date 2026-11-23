"""Minimal JAX/Flax trainer for functional autoencoders."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Callable, Sequence

import jax
import optax
from flax import serialization

from .checkpoint import write_json
from .core import TrainState


class Trainer:
    """Adam, cosine decay, clipping, and optional gradient accumulation.

    The trainer is model agnostic: it has no classifier schedules, parameter
    groups, warmup, or case-specific parameter handling. ``state.step`` counts
    microsteps; ``completed_updates`` counts Adam updates.
    """

    def __init__(
        self,
        autoencoder,
        loss_fn: Callable,
        train_loader,
        validation_loader=None,
        metrics: Sequence = (),
        *,
        accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        run_directory: str | Path | None = None,
    ):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        if len(train_loader) % accumulation_steps:
            raise ValueError("Each epoch must contain complete accumulated updates")
        self.autoencoder = autoencoder
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.metrics = tuple(metrics)
        self.accumulation_steps = int(accumulation_steps)
        self.gradient_clip = float(gradient_clip)
        self.run_directory = Path(run_directory) if run_directory is not None else None
        self.history = []
        self.metrics_history = {metric.name: [] for metric in self.metrics}
        self.training_loss_history = []
        self._learning_rate = None
        self._train_step = jax.jit(self._step)

    @property
    def train_step(self):
        """Compiled microstep, exposed for preflight and focused tests."""
        return self._train_step

    def initialize(
        self,
        key,
        *,
        learning_rate: float,
        minimum_learning_rate: float,
        total_updates: int,
    ) -> TrainState:
        """Initialize parameters and the complete optimizer state."""
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= minimum_learning_rate <= learning_rate:
            raise ValueError("Require 0 <= minimum_learning_rate <= learning_rate")
        if total_updates < 1:
            raise ValueError("total_updates must be positive")

        loader_epoch = getattr(self.train_loader, "epoch", None)
        u_dec, x_dec, u_enc, x_enc = next(iter(self.train_loader))
        if loader_epoch is not None:
            self.train_loader.epoch = loader_epoch
        _, init_key, state_key = jax.random.split(key, 3)
        variables = self.autoencoder.init(init_key, u_enc, x_enc, x_dec)

        self._learning_rate = optax.cosine_decay_schedule(
            init_value=float(learning_rate),
            decay_steps=int(total_updates),
            alpha=float(minimum_learning_rate) / float(learning_rate),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip),
            optax.adam(self._learning_rate),
        )
        if self.accumulation_steps > 1:
            optimizer = optax.MultiSteps(
                optimizer,
                every_k_schedule=self.accumulation_steps,
                use_grad_mean=True,
            )
        return TrainState.create(
            apply_fn=self.autoencoder.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats"),
            key=state_key,
        )

    def _step(self, state, batch):
        u_dec, x_dec, u_enc, x_enc = batch
        key, loss_key = jax.random.split(state.key)
        (loss, batch_stats), gradients = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(
            state.params,
            key=loss_key,
            batch_stats=state.batch_stats,
            u_enc=u_enc,
            x_enc=x_enc,
            u_dec=u_dec,
            x_dec=x_dec,
        )
        state = state.apply_gradients(grads=gradients)
        return loss, state.replace(batch_stats=batch_stats, key=key)

    def fit(
        self,
        state: TrainState,
        *,
        epochs: int,
        evaluation_interval: int = 1,
        start_epoch: int = 0,
    ):
        """Train from ``state`` and retain the lowest first validation metric."""
        if epochs <= start_epoch:
            raise ValueError("epochs must exceed start_epoch")
        if evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")
        if self._learning_rate is None:
            raise RuntimeError("initialize must be called before fit")
        if self.metrics and self.validation_loader is None:
            raise ValueError("validation_loader is required when metrics are supplied")

        self.history = []
        self.metrics_history = {metric.name: [] for metric in self.metrics}
        self.training_loss_history = []
        best_state = None
        best_metric = math.inf
        best_epoch = None
        started = time.perf_counter()

        for epoch in range(start_epoch + 1, epochs + 1):
            state, loss = self._train_epoch(state)
            microsteps = int(jax.device_get(state.step))
            completed_updates = microsteps // self.accumulation_steps
            row = {
                "epoch": epoch,
                "microsteps": microsteps,
                "completed_updates": completed_updates,
                "loss": loss,
                "learning_rate": float(jax.device_get(self._learning_rate(completed_updates))),
                "elapsed_seconds": time.perf_counter() - started,
            }

            if self.metrics and epoch % evaluation_interval == 0:
                validation = {}
                for metric in self.metrics:
                    value = float(metric(state, state.key, self.validation_loader))
                    self.metrics_history[metric.name].append(value)
                    validation[metric.name] = value
                row["validation"] = validation
                candidate = validation[self.metrics[0].name]
                if math.isfinite(candidate) and candidate < best_metric:
                    best_state = state
                    best_metric = candidate
                    best_epoch = epoch
                    self._save_best(state, epoch, validation)

            self.history.append(row)
            self._save_progress(state, row)
            print("EPOCH", json.dumps(row), flush=True)

        if best_state is None:
            best_state = state
            best_metric = None
            best_epoch = epochs
        return {
            "state": best_state,
            "last_state": state,
            "best_metric_name": self.metrics[0].name if self.metrics else None,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "training_loss_history": self.training_loss_history,
            "metrics_history": self.metrics_history,
            "history": self.history,
        }

    def _train_epoch(self, state):
        loss_sum = 0.0
        batches = 0
        for batch in self.train_loader:
            loss, state = self._train_step(state, batch)
            loss_sum = loss_sum + loss
            batches += 1
        if batches == 0:
            raise ValueError("train_loader produced no batches")
        loss = float(jax.device_get(loss_sum / batches))
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss: {loss}")
        self.training_loss_history.append(loss)
        return state, loss

    def _save_progress(self, state, row):
        if self.run_directory is None:
            return
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._save_state(state, "last.msgpack")
        write_json(self.run_directory / "history.json", self.history)
        write_json(self.run_directory / "status.json", {"status": "running", **row})

    def _save_best(self, state, epoch, validation):
        if self.run_directory is None:
            return
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._save_state(state, "best.msgpack")
        write_json(
            self.run_directory / "best.json",
            {"epoch": epoch, "validation": validation},
        )

    def _save_state(self, state, filename):
        path = self.run_directory / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(serialization.to_bytes(state))
        temporary.replace(path)
