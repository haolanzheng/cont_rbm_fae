"""Compatibility entry point; the maintained server implementation is train_jax."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('train_jax.py')), run_name='__main__')
