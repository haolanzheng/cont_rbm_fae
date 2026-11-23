"""Geometry, probability certificates, and exact assembled-PDE ROM identities."""

import numpy as np

from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.geometry_cpu import (
    CoordinateROM, ball_probability_lower, ball_step, search, sphere_dj,
    sphere_exp, uniform_parameters,
)


def test_round_derivative_matches_geodesic_directional_derivative():
    H = np.array([[2., .2, 0.], [.2, .8, .1], [0., .1, .2]])
    z = np.array([.3, -.4, .5])
    z /= np.linalg.norm(z)
    v = np.array([.1, .4, -.2])
    v -= (z @ v) * z
    derivative = sphere_dj(H, z)
    h = 1e-6
    forward, backward = sphere_exp(z, h*v), sphere_exp(z, -h*v)
    numerical = (forward @ H @ forward - backward @ H @ backward) / (2*h*np.linalg.norm(H))
    np.testing.assert_allclose(derivative @ v, numerical, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(derivative @ z, 0, atol=1e-15)


def test_exp_update_feasible_and_stationary_start_fixed():
    H = np.diag([2., 1., .2, 0.])
    initial = uniform_parameters(np.random.default_rng(2), 128, 4, "sphere")
    best, data = search(H, initial, 100)
    np.testing.assert_allclose(np.linalg.norm(best, axis=1), 1, atol=5e-15)
    assert data["minimum_objective_increment"] >= -1e-14
    fixed, _ = search(H, np.array([0., 1., 0., 0.]), 100)
    np.testing.assert_array_equal(fixed, [0., 1., 0., 0.])


def test_relaxation_distinguishes_trial_activity_from_boundary_contact():
    H = np.diag([2., 1., 0.])
    z = np.array([[.9, 0., 0.]])
    relaxed, active = ball_step(H, z, .5)
    assert active[0]
    assert .9 < np.linalg.norm(relaxed) < 1
    hard, _ = ball_step(H, z, 1.)
    np.testing.assert_allclose(np.linalg.norm(hard), 1)


def test_hard_ball_matches_independent_closed_formula():
    H = np.array([[2., .2, 0.], [.2, .5, .05], [0., .05, .1]])
    initial = np.array([.12, -.31, .23])
    M = np.eye(3) + H / np.linalg.norm(H)
    for count in (0, 1, 5, 25):
        raw = np.linalg.matrix_power(M, count) @ initial
        exact = raw / max(1., np.linalg.norm(raw))
        best, _ = search(H, initial, count, "ball", omega=1.)
        np.testing.assert_allclose(best, exact, rtol=1e-12, atol=1e-14)


def test_ball_certificate_event_implies_success_for_coupled_psd_with_kernel():
    rng = np.random.default_rng(21)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    H = orthogonal @ np.diag([1., .6, .3, .1, .02, 0., 0., 0.]) @ orthogonal.T
    z = uniform_parameters(rng, 4096, 8, "ball")
    gamma, count, omega = .9, 25, .5
    a = 1 + 1/np.linalg.norm(H)
    b = np.sqrt(1 + gamma**2*(a*a - 1))
    c = 1-omega+omega*a/b
    event = (z @ orthogonal[:, 0])**2 >= gamma**2*c**(-2*count)
    best, _ = search(H, z, count, "ball", omega=omega)
    assert event.sum() > 500
    assert np.all(np.sum(best[event]*(best[event] @ H.T), axis=1) >= gamma**2 - 1e-13)
    bounds = ball_probability_lower(H, gamma, np.array([0, 1, 5, 25, 100]), omega)
    assert np.all(np.diff(bounds) > 0)


def test_uniform_ball_radial_distribution():
    z = uniform_parameters(np.random.default_rng(90), 50000, 8, "ball")
    radial_cdf = np.linalg.norm(z, axis=1)**8
    assert abs(radial_cdf.mean() - .5) < .005
    assert abs((radial_cdf <= .25).mean() - .25) < .007
    assert abs(np.mean(np.sum(z*z, axis=1)) - 8/10) < .005


def test_coordinate_rom_matches_full_fem_residual_and_error():
    problem = FEMProblem.build(4, 8, element_order=3)
    G0 = problem.solution_gram
    H0 = problem.B.T @ problem.solve_X(problem.B)
    rom = CoordinateROM(G0, H0)
    starts = uniform_parameters(np.random.default_rng(44), 4, 8, "sphere")
    for z in starts:
        V = problem.S @ rom.Q
        C = V.T @ problem.B
        R = problem.B - (problem.A @ V) @ C
        full_H = R.T @ problem.solve_X(R)
        full_G = G0 - C.T @ C
        H, G = rom.matrices()
        np.testing.assert_allclose(H, full_H, atol=2e-13, rtol=1e-11)
        np.testing.assert_allclose(G, full_G, atol=2e-13, rtol=1e-11)
        rom.append(z)
    assert rom.orthogonality_defect() < 1e-12
