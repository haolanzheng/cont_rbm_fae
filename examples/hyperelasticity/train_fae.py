"""Train the widened JAX FAE from scratch on the fixed function-field dataset."""

import argparse
import fcntl
from pathlib import Path

from contgreedy.data import load_config
from contgreedy.fae.training import train

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/fae128_jax_w32.json",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(
            config,
            data_directory=ROOT / "data" / config["data_directory"],
            run_directory=args.run_dir,
            resume=args.resume,
            stop_after_epochs=args.stop_after_epochs,
            require_gpu=True,
        )


if __name__ == "__main__":
    main()
