"""Checks for the compressed FE and distribution-mixture experiment."""

import numpy as np

from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.distribution_tradeoff import (
    append_g_orthonormal,
    mixture_pod,
    residual_gram,
)


def test_parameter_space_residual_and_error_match_direct_fe():
    problem = FEMProblem.build(3, 4, element_order=3)
    G = problem.solution_gram
    Q = append_g_orthonormal(
        G, np.empty((problem.d, 0)), np.array([0.4, -0.2, 0.8, 0.3])
    )
    V = problem.S @ Q
    R = problem.B - (problem.A @ V) @ (V.T @ problem.B)
    XR = np.column_stack([problem.solve_X(R[:, j]) for j in range(problem.d)])
    direct_indicator = R.T @ XR
    XB = np.column_stack([problem.solve_X(problem.B[:, j]) for j in range(problem.d)])
    K = problem.B.T @ XB
    projection = np.eye(problem.d) - Q @ (Q.T @ G)
    compressed_indicator = projection.T @ K @ projection
    np.testing.assert_allclose(
        compressed_indicator, direct_indicator, rtol=2e-11, atol=2e-12
    )
    energy_residual = problem.S - V @ (V.T @ problem.B)
    direct_error = energy_residual.T @ (problem.A @ energy_residual)
    np.testing.assert_allclose(
        residual_gram(G, Q), direct_error, rtol=2e-11, atol=2e-12
    )


def test_mixture_pod_switches_from_typical_to_worst_modes():
    G = np.diag([1.0, 0.6, 0.3, 0.1])
    empirical_covariance = np.diag([0.01, 0.01, 0.7, 0.28])
    typical = mixture_pod(G, empirical_covariance, 0.0, 2)
    worst = mixture_pod(G, empirical_covariance, 1.0, 2)
    np.testing.assert_allclose(typical.T @ G @ typical, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(worst.T @ G @ worst, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        np.diag(residual_gram(G, typical)),
        [1.0, 0.6, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.diag(residual_gram(G, worst)),
        [0.0, 0.0, 0.3, 0.1],
        atol=1e-12,
    )
