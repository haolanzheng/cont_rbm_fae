"""Truth solver contract; assembly and solver choices belong to each example."""

from typing import Protocol


class TruthSolver(Protocol):
    def fom(self, parameter):
        """Return one truth state. Transient examples may return a trajectory."""
        ...
