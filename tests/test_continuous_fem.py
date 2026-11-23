"""Checks for mesh-independent loads and conforming-FE truth projections."""

import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
from skfem import Basis, ElementTriP1, LinearForm, MeshTri, asm

from examples.baseline.cli import run_config
from examples.baseline.continuous_fem import (
    FEMProblem,
    continuum_energy_gram,
    sine_pairs,
)
from examples.baseline.rom import matrices


def test_continuum_construction_is_nested_and_orthonormal():
    p4 = FEMProblem.build(8, 4, element_order=1)
    p8 = FEMProblem.build(8, 8, element_order=1)
    assert p4.modes == p8.modes[:4]
    np.testing.assert_allclose(
        p4.source_transform, p8.source_transform[:4, :4], atol=1e-13
    )
    g = continuum_energy_gram(sine_pairs(8), 0.25)
    np.testing.assert_allclose(
        p8.source_transform.T @ g @ p8.source_transform, np.eye(8), atol=2e-12
    )


def test_analytic_sources_match_weak_fem_assembly():
    p = FEMProblem.build(8, 4, element_order=1)
    coords = np.linspace(0.0, 1.0, p.n_grid + 2)
    basis = Basis(
        MeshTri.init_tensor(coords, coords), ElementTriP1(), intorder=p.quadrature_order
    )
    interior = basis.complement_dofs(basis.get_dofs())

    def make_load(j):
        @LinearForm
        def load(v, w):
            return p.source_value(j, w.x[0], w.x[1]) * v

        return load

    for j in range(p.d):
        direct = asm(make_load(j), basis)[interior]
        np.testing.assert_allclose(direct, p.B[:, j], rtol=2e-11, atol=2e-11)


def test_fem_projection_and_rom_bounds():
    coarse = FEMProblem.build(6, 4, element_order=2)
    fine = FEMProblem.build(10, 4, element_order=2)
    assert fine.projection_error < coarse.projection_error
    d = fine.spectrum_ratio ** np.arange(fine.d)
    normalized = fine.solution_gram / np.outer(d, d)
    assert la.eigvalsh(np.eye(fine.d) - normalized)[0] > -1e-10
    assert la.eigvalsh(normalized)[0] >= 1 - fine.projection_error**2 - 1e-10
    H, G, _ = matrices(fine, np.empty((fine.dofs, 0)))
    np.testing.assert_allclose(G, fine.solution_gram, atol=1e-12)
    # Uniform ellipticity gives the residual/energy equivalence.
    assert la.eigvalsh(H - G / (1 + fine.contrast))[0] > -1e-9
    assert la.eigvalsh(G / (1 - fine.contrast) - H)[0] > -1e-9


def test_continuous_fe_cli_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "examples/baseline/configs/continuous_fe_smoke.json").read_text()
    )
    result = run_config(config, tmp_path / "fe_smoke")
    assert result["hf_diagnostics"]["element_order"] == 3
    assert result["hf_diagnostics"]["max_relative_energy_projection_error"] < 0.05
    assert len(result["runs"]) == 3
