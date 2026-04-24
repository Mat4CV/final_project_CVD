import numpy as np

from src.plane_scoring_detection.detection import FourierMotionDetector, FourierMotionConfig


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


def nearest_detected_velocity(result, gt):
    if len(result.detected_velocities) == 0:
        return None, np.inf

    gt = np.asarray(gt, dtype=np.float32)
    d = np.linalg.norm(result.detected_velocities - gt[None, :], axis=1)
    idx = np.argmin(d)

    return result.detected_velocities[idx], d[idx]


def test_detector_single_gaussian_motion():
    gt = (1.0, 0.0)

    video = make_moving_gaussian(
        T=32,
        H=64,
        W=64,
        velocity=gt,
        center=(16.0, 32.0),
    )

    config = FourierMotionConfig(
        velocity_bounds=(-2.0, 2.0, 101),
        sigma=0.5,
        alpha=0.01,
        dc_bins=1,
        keep_frac=0.1,
        use_gpu=False,
        use_hann_window=True,
        max_detections=5,
        min_detection_separation=2,
        verbose=False,
    )

    detector = FourierMotionDetector(config)
    result = detector.detect(video)

    v = result.velocity_grid
    imax, jmax = np.unravel_index(np.argmax(result.energies), result.energies.shape)
    energy_argmax = np.array([v[imax], v[jmax]])

    pred, dist = nearest_detected_velocity(result, gt)

    assert pred is not None
    assert dist <= 0.2


def test_detector_diagonal_motion():
    gt = (1.0, 0.5)

    video = make_moving_gaussian(
        T=32,
        H=64,
        W=64,
        velocity=gt,
        center=(16.0, 16.0),
    )

    config = FourierMotionConfig(
        velocity_bounds=(-2.0, 2.0, 101),
        sigma=0.5,
        alpha=0.01,
        dc_bins=1,
        keep_frac=0.1,
        use_gpu=False,
        use_hann_window=True,
        max_detections=5,
        min_detection_separation=2,
        verbose=False,
    )

    detector = FourierMotionDetector(config)
    result = detector.detect(video)

    v = result.velocity_grid
    imax, jmax = np.unravel_index(np.argmax(result.energies), result.energies.shape)
    energy_argmax = np.array([v[imax], v[jmax]])

    pred, dist = nearest_detected_velocity(result, gt)

    assert pred is not None
    assert dist <= 0.25


def test_detector_result_shapes():
    video = make_moving_gaussian(T=16, H=32, W=32, velocity=(1.0, 0.0))

    config = FourierMotionConfig(
        velocity_bounds=(-2.0, 2.0, 17),
        sigma=0.5,
        alpha=0.01,
        keep_frac=0.2,
        use_gpu=False,
        verbose=False,
    )

    result = FourierMotionDetector(config).detect(video)
    

    assert result.energies.shape == (17, 17)
    assert result.detections.shape == (17, 17)
    assert result.velocity_grid.shape == (17,)
    assert result.F.shape == video.shape
    assert result.ft.shape == (16,)
    assert result.fy.shape == (32,)
    assert result.fx.shape == (32,)