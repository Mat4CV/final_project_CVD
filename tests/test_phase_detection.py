import itertools

import numpy as np
import pytest

from src.phase_model.detection import (
    angular_distance,
    expected_phase_step,
    frequency_grid,
    phase_hough_votes,
    reconstruct_from_spectra,
    solve_phasors_prony,
    solve_two_phasors_closed_form,
    sort_components_by_velocity,
    top_k_peaks,
)


def test_frequency_grid_shape_and_zero_frequency():
    H, W = 12, 18
    wx, wy = frequency_grid(H, W)

    assert wx.shape == (H, W)
    assert wy.shape == (H, W)

    assert wx[0, 0] == pytest.approx(0.0)
    assert wy[0, 0] == pytest.approx(0.0)


def test_frequency_grid_x_varies_along_columns_y_varies_along_rows():
    H, W = 8, 10
    wx, wy = frequency_grid(H, W)

    # wx should be constant down rows for a fixed column.
    assert np.allclose(wx[0, :], wx[3, :])
    assert np.allclose(wx[2, :], wx[7, :])

    # wy should be constant across columns for a fixed row.
    assert np.allclose(wy[:, 0], wy[:, 5])
    assert np.allclose(wy[:, 2], wy[:, 9])

    # wx changes along columns.
    assert not np.allclose(wx[0, :], wx[0, 0])

    # wy changes along rows.
    assert not np.allclose(wy[:, 0], wy[0, 0])


def test_angular_distance_basic_values():
    assert angular_distance(0.0, 0.0) == pytest.approx(0.0)
    assert angular_distance(0.0, np.pi) == pytest.approx(np.pi)
    assert angular_distance(np.pi, -np.pi) == pytest.approx(0.0)
    assert angular_distance(0.0, 2.0 * np.pi) == pytest.approx(0.0)


def test_angular_distance_vectorized():
    a = np.array([0.0, np.pi, -np.pi, 0.1])
    b = np.array([0.0, -np.pi, np.pi, 0.2])

    d = angular_distance(a, b)

    assert d.shape == a.shape
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(0.0)
    assert d[2] == pytest.approx(0.0)
    assert d[3] == pytest.approx(0.1)


def test_expected_phase_step_unit_magnitude(small_frequency_grid):
    _, _, wx, wy = small_frequency_grid

    z = expected_phase_step(wx, wy, velocity=(1.25, -0.5), dt=1.0)

    assert z.shape == wx.shape
    assert np.allclose(np.abs(z), 1.0)


def test_solve_two_phasors_closed_form_scalar_exact():
    A_true = 2.0 + 1.0j
    B_true = -0.5 + 0.7j

    z1_true = np.exp(1j * 0.35)
    z2_true = np.exp(1j * -0.9)

    samples = []
    for t in range(4):
        samples.append(A_true * z1_true**t + B_true * z2_true**t)

    F_seq = np.asarray(samples, dtype=np.complex128).reshape(4, 1, 1)

    z, A, valid = solve_two_phasors_closed_form(F_seq)

    assert valid.shape == (1, 1)
    assert valid[0, 0]

    recovered = [
        (z[0, 0, 0], A[0, 0, 0], z[1, 0, 0], A[1, 0, 0]),
        (z[1, 0, 0], A[1, 0, 0], z[0, 0, 0], A[0, 0, 0]),
    ]

    ok = False
    for za, Aa, zb, Bb in recovered:
        if (
            np.allclose(za, z1_true, atol=1e-10)
            and np.allclose(zb, z2_true, atol=1e-10)
            and np.allclose(Aa, A_true, atol=1e-10)
            and np.allclose(Bb, B_true, atol=1e-10)
        ):
            ok = True

    assert ok


def test_solve_two_phasors_closed_form_requires_3d_input():
    F_seq = np.zeros((4, 8), dtype=np.complex128)

    with pytest.raises(ValueError, match="F_seq must have shape"):
        solve_two_phasors_closed_form(F_seq)


def test_solve_two_phasors_closed_form_requires_four_frames():
    F_seq = np.zeros((3, 8, 8), dtype=np.complex128)

    with pytest.raises(ValueError, match="requires at least four frames"):
        solve_two_phasors_closed_form(F_seq)


def test_solve_phasors_prony_m1_scalar_exact():
    A_true = 3.0 - 0.2j
    z_true = np.exp(1j * 0.6)

    T = 4
    samples = np.asarray(
        [A_true * z_true**t for t in range(T)],
        dtype=np.complex128,
    )

    F_seq = samples.reshape(T, 1, 1)

    z, A, valid = solve_phasors_prony(F_seq, num_components=1)

    assert z.shape == (1, 1, 1)
    assert A.shape == (1, 1, 1)
    assert valid[0, 0]

    assert np.allclose(z[0, 0, 0], z_true, atol=1e-10)
    assert np.allclose(A[0, 0, 0], A_true, atol=1e-10)


def test_solve_phasors_prony_m2_scalar_exact():
    amps_true = np.array([1.2 + 0.3j, -0.7 + 0.9j])
    roots_true = np.array([
        np.exp(1j * 0.25),
        np.exp(1j * -0.8),
    ])

    T = 6
    samples = np.zeros(T, dtype=np.complex128)

    for t in range(T):
        samples[t] = np.sum(amps_true * roots_true**t)

    F_seq = samples.reshape(T, 1, 1)

    z, A, valid = solve_phasors_prony(F_seq, num_components=2)

    assert valid[0, 0]

    # Root order is arbitrary.
    recovered_pairs = [(z[i, 0, 0], A[i, 0, 0]) for i in range(2)]

    ok = False
    for perm in itertools.permutations(range(2)):
        local_ok = True

        for recovered_idx, true_idx in enumerate(perm):
            zr, Ar = recovered_pairs[recovered_idx]
            zt = roots_true[true_idx]
            At = amps_true[true_idx]

            local_ok &= np.allclose(zr, zt, atol=1e-8)
            local_ok &= np.allclose(Ar, At, atol=1e-8)

        ok |= local_ok

    assert ok


def test_solve_phasors_prony_m3_scalar_exact():
    amps_true = np.array([
        1.0 + 0.1j,
        -0.4 + 0.8j,
        0.7 - 0.3j,
    ])

    roots_true = np.array([
        np.exp(1j * 0.2),
        np.exp(1j * -0.7),
        np.exp(1j * 1.1),
    ])

    T = 8
    samples = np.zeros(T, dtype=np.complex128)

    for t in range(T):
        samples[t] = np.sum(amps_true * roots_true**t)

    F_seq = samples.reshape(T, 1, 1)

    z, A, valid = solve_phasors_prony(F_seq, num_components=3)

    assert valid[0, 0]

    recovered_roots = z[:, 0, 0]
    recovered_amps = A[:, 0, 0]

    ok = False

    for perm in itertools.permutations(range(3)):
        local_ok = True

        for recovered_idx, true_idx in enumerate(perm):
            local_ok &= np.allclose(
                recovered_roots[recovered_idx],
                roots_true[true_idx],
                atol=1e-7,
            )
            local_ok &= np.allclose(
                recovered_amps[recovered_idx],
                amps_true[true_idx],
                atol=1e-7,
            )

        ok |= local_ok

    assert ok


def test_solve_phasors_prony_requires_3d_input():
    F_seq = np.zeros((8, 8), dtype=np.complex128)

    with pytest.raises(ValueError, match="F_seq must have shape"):
        solve_phasors_prony(F_seq, num_components=2)


def test_solve_phasors_prony_requires_enough_frames():
    F_seq = np.zeros((5, 4, 4), dtype=np.complex128)

    with pytest.raises(ValueError, match="Need at least 2m frames"):
        solve_phasors_prony(F_seq, num_components=3)


def test_solve_phasors_prony_rejects_invalid_num_components():
    F_seq = np.zeros((4, 4, 4), dtype=np.complex128)

    with pytest.raises(ValueError, match="num_components"):
        solve_phasors_prony(F_seq, num_components=0)


def test_phase_hough_votes_detects_known_single_velocity():
    H, W = 32, 32
    wx, wy = frequency_grid(H, W)

    true_velocity = (1.0, -0.5)
    z = expected_phase_step(wx, wy, true_velocity, dt=1.0)

    z_fields = z[None, :, :]
    weights = np.ones_like(z_fields, dtype=np.float64)

    valid = np.ones((H, W), dtype=bool)
    valid[0, 0] = False

    vx_values = np.linspace(-2.0, 2.0, 81)
    vy_values = np.linspace(-2.0, 2.0, 81)

    accumulator = phase_hough_votes(
        z_fields=z_fields,
        wx=wx,
        wy=wy,
        valid=valid,
        vx_values=vx_values,
        vy_values=vy_values,
        weights=weights,
        sigma=0.05,
    )

    velocities = top_k_peaks(
        accumulator,
        vx_values=vx_values,
        vy_values=vy_values,
        k=1,
        min_separation=5,
    )

    assert len(velocities) == 1
    vx_hat, vy_hat = velocities[0]

    assert vx_hat == pytest.approx(true_velocity[0], abs=0.06)
    assert vy_hat == pytest.approx(true_velocity[1], abs=0.06)


def test_phase_hough_votes_supports_multiple_branches(match_velocity_sets_fn):
    H, W = 32, 32
    wx, wy = frequency_grid(H, W)

    true_velocities = [
        (0.75, 0.25),
        (-1.0, 0.5),
        (1.25, -0.75),
    ]

    z_fields = np.stack(
        [expected_phase_step(wx, wy, v) for v in true_velocities],
        axis=0,
    )

    weights = np.ones_like(z_fields, dtype=np.float64)

    valid = np.ones((H, W), dtype=bool)
    valid[0, 0] = False

    vx_values = np.linspace(-2.0, 2.0, 81)
    vy_values = np.linspace(-2.0, 2.0, 81)

    accumulator = phase_hough_votes(
        z_fields=z_fields,
        wx=wx,
        wy=wy,
        valid=valid,
        vx_values=vx_values,
        vy_values=vy_values,
        weights=weights,
        sigma=0.05,
    )

    estimated = top_k_peaks(
        accumulator,
        vx_values=vx_values,
        vy_values=vy_values,
        k=3,
        min_separation=5,
    )

    assert match_velocity_sets_fn(estimated, true_velocities, tolerance=0.08)


def test_phase_hough_votes_validates_shapes():
    H, W = 8, 8
    wx, wy = frequency_grid(H, W)
    z_fields = np.ones((2, H, W), dtype=np.complex128)
    valid = np.ones((H, W), dtype=bool)
    vx_values = np.linspace(-1, 1, 5)
    vy_values = np.linspace(-1, 1, 5)

    with pytest.raises(ValueError, match="weights"):
        phase_hough_votes(
            z_fields=z_fields,
            wx=wx,
            wy=wy,
            valid=valid,
            vx_values=vx_values,
            vy_values=vy_values,
            weights=np.ones((H, W)),
        )


def test_top_k_peaks_returns_expected_order_and_suppresses_neighbors():
    accumulator = np.zeros((10, 10), dtype=float)
    accumulator[2, 3] = 10.0
    accumulator[2, 4] = 9.0
    accumulator[8, 8] = 7.0

    vx_values = np.arange(10, dtype=float)
    vy_values = np.arange(10, dtype=float)

    peaks = top_k_peaks(
        accumulator,
        vx_values=vx_values,
        vy_values=vy_values,
        k=2,
        min_separation=1,
    )

    assert peaks == [(3.0, 2.0), (8.0, 8.0)]


def test_top_k_peaks_validates_accumulator_shape():
    accumulator = np.zeros((4, 5))
    vx_values = np.arange(6)
    vy_values = np.arange(4)

    with pytest.raises(ValueError, match="accumulator shape"):
        top_k_peaks(accumulator, vx_values, vy_values, k=1)


def test_sort_components_by_velocity_m2_exact():
    H, W = 8, 8
    wx, wy = frequency_grid(H, W)

    velocities = [
        (1.0, 0.0),
        (-0.5, 0.75),
    ]

    z_obj1 = expected_phase_step(wx, wy, velocities[0])
    z_obj2 = expected_phase_step(wx, wy, velocities[1])

    F_obj1 = np.ones((H, W), dtype=np.complex128) * (2.0 + 0.5j)
    F_obj2 = np.ones((H, W), dtype=np.complex128) * (-1.0 + 0.25j)

    z = np.zeros((2, H, W), dtype=np.complex128)
    A = np.zeros((2, H, W), dtype=np.complex128)

    checker = np.indices((H, W)).sum(axis=0) % 2 == 0

    z[0, checker] = z_obj1[checker]
    A[0, checker] = F_obj1[checker]
    z[1, checker] = z_obj2[checker]
    A[1, checker] = F_obj2[checker]

    z[0, ~checker] = z_obj2[~checker]
    A[0, ~checker] = F_obj2[~checker]
    z[1, ~checker] = z_obj1[~checker]
    A[1, ~checker] = F_obj1[~checker]

    # Only test frequencies where the two velocities induce distinguishable phases.
    phase_sep = angular_distance(np.angle(z_obj1), np.angle(z_obj2))
    valid = phase_sep > 1e-6

    spectra = sort_components_by_velocity(
        A=A,
        z=z,
        velocities=velocities,
        wx=wx,
        wy=wy,
        valid=valid,
    )

    assert len(spectra) == 2

    assert np.allclose(spectra[0][valid], F_obj1[valid])
    assert np.allclose(spectra[1][valid], F_obj2[valid])

    # Invalid/ambiguous frequencies are left at zero by construction.
    assert np.allclose(spectra[0][~valid], 0.0)
    assert np.allclose(spectra[1][~valid], 0.0)


def test_sort_components_by_velocity_m3_exact():
    H, W = 6, 6
    wx, wy = frequency_grid(H, W)

    velocities = [
        (1.0, 0.0),
        (-0.5, 0.5),
        (0.25, -1.0),
    ]

    true_spectra = [
        np.full((H, W), 1.0 + 0.1j),
        np.full((H, W), 2.0 - 0.3j),
        np.full((H, W), -0.5 + 0.7j),
    ]

    true_z = [expected_phase_step(wx, wy, v) for v in velocities]

    z = np.zeros((3, H, W), dtype=np.complex128)
    A = np.zeros((3, H, W), dtype=np.complex128)

    perms = list(itertools.permutations(range(3)))

    for y in range(H):
        for x in range(W):
            perm = perms[(x + y) % len(perms)]

            for branch, obj in enumerate(perm):
                z[branch, y, x] = true_z[obj][y, x]
                A[branch, y, x] = true_spectra[obj][y, x]

    valid = np.ones((H, W), dtype=bool)

    spectra = sort_components_by_velocity(
        A=A,
        z=z,
        velocities=velocities,
        wx=wx,
        wy=wy,
        valid=valid,
    )

    assert len(spectra) == 3

    for got, expected in zip(spectra, true_spectra):
        assert np.allclose(got, expected)


def test_sort_components_by_velocity_requires_matching_shapes():
    A = np.zeros((2, 8, 8), dtype=np.complex128)
    z = np.zeros((2, 8, 7), dtype=np.complex128)
    wx = np.zeros((8, 8))
    wy = np.zeros((8, 8))
    valid = np.ones((8, 8), dtype=bool)

    with pytest.raises(ValueError, match="same shape"):
        sort_components_by_velocity(
            A=A,
            z=z,
            velocities=[(0, 0), (1, 0)],
            wx=wx,
            wy=wy,
            valid=valid,
        )


def test_reconstruct_from_spectra_roundtrip():
    rng = np.random.default_rng(0)

    img1 = rng.normal(size=(16, 16))
    img2 = rng.normal(size=(16, 16))

    spectra = [np.fft.fft2(img1), np.fft.fft2(img2)]

    reconstructed = reconstruct_from_spectra(spectra)

    assert len(reconstructed) == 2
    assert np.allclose(reconstructed[0], img1)
    assert np.allclose(reconstructed[1], img2)