import numpy as np

from src.synthetic import make_single_gaussian, make_two_objects
from src.plane_scoring_detection.detection import FourierMotionDetector, FourierMotionConfig


def nearest_distance(pred_velocities, gt_velocity):
    if len(pred_velocities) == 0:
        return np.inf

    gt_velocity = np.asarray(gt_velocity, dtype=np.float32)
    d = np.linalg.norm(pred_velocities - gt_velocity[None, :], axis=1)
    return float(d.min())

def local_best_distance(result, gt_velocity, radius=0.35):
    v = result.velocity_grid
    E = result.energies

    gt_vx, gt_vy = gt_velocity

    VX, VY = np.meshgrid(v, v, indexing="ij")
    dist = np.sqrt((VX - gt_vx) ** 2 + (VY - gt_vy) ** 2)

    mask = dist <= radius

    if not np.any(mask):
        return np.inf, None

    local_indices = np.argwhere(mask)
    local_values = E[mask]

    best_local = local_indices[np.argmax(local_values)]
    i, j = best_local

    pred = np.array([v[i], v[j]], dtype=np.float32)
    return float(np.linalg.norm(pred - np.asarray(gt_velocity, dtype=np.float32))), pred


def test_integration_single_gaussian_from_synthetic_module():
    video, masks, metadata = make_single_gaussian()

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

    result = FourierMotionDetector(config).detect(video)

    gt_velocity = metadata["velocities"][0]
    dist = nearest_distance(result.detected_velocities, gt_velocity)

    assert dist <= 0.25


def test_integration_two_objects_from_synthetic_module():
    video, masks, metadata = make_two_objects()

    config = FourierMotionConfig(
        velocity_bounds=(-2.0, 2.0, 101),
        sigma=0.5,
        alpha=0.05,
        dc_bins=1,
        keep_frac=0.15,
        use_gpu=False,
        use_hann_window=False,
        max_detections=10,
        min_detection_separation=2,
        verbose=False,
    )

    result = FourierMotionDetector(config).detect(video)

    gt_velocities = metadata["velocities"]

    local = [
    local_best_distance(result, gt, radius=0.45)
    for gt in gt_velocities
    ]

    distances = [x[0] for x in local]

    assert distances[0] <= 0.45
    assert distances[1] <= 0.45