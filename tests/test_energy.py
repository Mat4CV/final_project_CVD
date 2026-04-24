import numpy as np
import pytest
import torch

from src.plane_scoring_detection.fft import fft3_video, preprocess_power_spectrum
from src.plane_scoring_detection.energy import (
    VelocityGrid,
    VelocityPlaneScorer,
    VelocityPlaneScorerConfig,
    normalize_velocity,
)


def make_moving_gaussian(
    T=32,
    H=64,
    W=64,
    velocity=(1.0, 0.0),
    center=(16.0, 32.0),
    sigma=4.0,
):
    vx, vy = velocity
    cx0, cy0 = center

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    video = np.zeros((T, H, W), dtype=np.float32)

    for t in range(T):
        cx = cx0 + vx * t
        cy = cy0 + vy * t
        video[t] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    return video


def test_normalize_velocity():
    v = np.array([1.0, 2.0], dtype=np.float32)

    out = normalize_velocity(v, num_t=32, num_space=64)

    assert np.allclose(out, np.array([0.5, 1.0]))


def test_velocity_grid_values():
    grid = VelocityGrid(-2.0, 2.0, 5)

    assert np.allclose(grid.values(), np.array([-2, -1, 0, 1, 2], dtype=np.float32))


def test_velocity_plane_scorer_output_shape_numpy():
    video = make_moving_gaussian(T=16, H=32, W=32, velocity=(1.0, 0.0))

    F, ft, fy, fx = fft3_video(video, dt=1.0, shift=True)
    power = preprocess_power_spectrum(F, ft, dc_bins=1)

    grid = VelocityGrid(-2.0, 2.0, 17)
    config = VelocityPlaneScorerConfig(
        backend="numpy",
        keep_frac=0.2,
        sigma=0.75,
        verbose=False,
    )
    scorer = VelocityPlaneScorer(grid, config)

    E = scorer.compute(power, ft, fy, fx)

    assert E.shape == (17, 17)
    assert np.all(np.isfinite(E))
    assert np.all(E >= 0)


def test_velocity_plane_scorer_detects_known_x_motion_numpy():
    gt_vx = 1.0
    gt_vy = 0.0

    video = make_moving_gaussian(
        T=32,
        H=64,
        W=64,
        velocity=(gt_vx, gt_vy),
        center=(16.0, 32.0),
    )

    F, ft, fy, fx = fft3_video(video, dt=1.0, shift=True)
    power = preprocess_power_spectrum(F, ft, dc_bins=1)

    grid = VelocityGrid(-2.0, 2.0, 41)
    scorer = VelocityPlaneScorer(
        grid,
        VelocityPlaneScorerConfig(
            backend="numpy",
            keep_frac=0.1,
            sigma=0.75,
            verbose=False,
        ),
    )

    E = scorer.compute(power, ft, fy, fx)

    v = grid.values()
    i, j = np.unravel_index(np.argmax(E), E.shape)

    pred_vx = v[i]
    pred_vy = v[j]

    assert abs(pred_vx - gt_vx) <= 0.15
    assert abs(pred_vy - gt_vy) <= 0.15


def test_velocity_plane_scorer_detects_known_diagonal_motion_numpy():
    gt_vx = 1.0
    gt_vy = 0.5

    video = make_moving_gaussian(
        T=32,
        H=64,
        W=64,
        velocity=(gt_vx, gt_vy),
        center=(16.0, 16.0),
    )

    F, ft, fy, fx = fft3_video(video, dt=1.0, shift=True)
    power = preprocess_power_spectrum(F, ft, dc_bins=1)

    grid = VelocityGrid(-2.0, 2.0, 41)
    scorer = VelocityPlaneScorer(
        grid,
        VelocityPlaneScorerConfig(
            backend="numpy",
            keep_frac=0.1,
            sigma=0.75,
            verbose=False,
        ),
    )

    E = scorer.compute(power, ft, fy, fx)

    v = grid.values()
    i, j = np.unravel_index(np.argmax(E), E.shape)

    pred_vx = v[i]
    pred_vy = v[j]

    assert abs(pred_vx - gt_vx) <= 0.15
    assert abs(pred_vy - gt_vy) <= 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_velocity_plane_scorer_torch_matches_numpy_approximately():
    video = make_moving_gaussian(T=16, H=32, W=32, velocity=(1.0, 0.5))

    F, ft, fy, fx = fft3_video(video, dt=1.0, shift=True)
    power = preprocess_power_spectrum(F, ft, dc_bins=1)

    grid = VelocityGrid(-2.0, 2.0, 17)

    scorer_np = VelocityPlaneScorer(
        grid,
        VelocityPlaneScorerConfig(
            backend="numpy",
            keep_frac=0.2,
            sigma=0.75,
            verbose=False,
        ),
    )

    scorer_torch = VelocityPlaneScorer(
        grid,
        VelocityPlaneScorerConfig(
            backend="torch",
            device="cuda",
            keep_frac=0.2,
            sigma=0.75,
            verbose=False,
        ),
    )

    E_np = scorer_np.compute(power, ft, fy, fx)
    E_torch = scorer_torch.compute(power, ft, fy, fx)

    E_np = E_np / (E_np.max() + 1e-8)
    E_torch = E_torch / (E_torch.max() + 1e-8)

    assert np.allclose(E_np, E_torch, atol=1e-4, rtol=1e-3)