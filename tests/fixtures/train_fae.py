"""Resume the exact input-only run specified by tab:fae-config in the manuscript."""

import argparse
from pathlib import Path
from contgreedy.data import load_config
from contgreedy.fae.training import train

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/fae128.json"
    )
    parser.add_argument("--tag", default="fae128")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--stop-after-epochs", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    train(
        config,
        data_directory=args.data_dir or ROOT / "data" / config["data_directory"],
        run_directory=args.run_dir or ROOT / "runs" / args.tag,
        resume=args.resume,
        stop_after_epochs=args.stop_after_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
