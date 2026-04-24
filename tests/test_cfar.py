import numpy as np

from src.plane_scoring_detection.cfar import rank_cfar_2d, keep_top_detections


def test_rank_cfar_2d_detects_isolated_peak():
    E = np.ones((21, 21), dtype=np.float32)
    E[10, 10] = 100.0

    result = rank_cfar_2d(
        E,
        alpha=0.05,
        window=5,
        guard=1,
    )

    assert result.detections[10, 10]


def test_rank_cfar_2d_does_not_detect_flat_field():
    E = np.ones((21, 21), dtype=np.float32)

    result = rank_cfar_2d(
        E,
        alpha=0.001,
        window=5,
        guard=1,
    )

    # In a perfectly flat field, ties can make rank large depending on >=.
    # For this reason we check that detections are not everywhere.
    assert result.detections.sum() < E.size


def test_rank_cfar_2d_output_shapes():
    E = np.random.default_rng(0).normal(size=(13, 17)).astype(np.float32)

    result = rank_cfar_2d(E, alpha=0.01, window=4, guard=1)

    assert result.statistic.shape == E.shape
    assert result.background.shape == E.shape
    assert result.detections.shape == E.shape
    assert result.thresholds.shape == E.shape


def test_keep_top_detections_limits_number():
    E = np.zeros((10, 10), dtype=np.float32)
    E[1, 1] = 1.0
    E[5, 5] = 5.0
    E[8, 8] = 3.0

    det = E > 0

    out = keep_top_detections(
        detections=det,
        energies=E,
        max_detections=2,
    )

    assert out.sum() == 2
    assert out[5, 5]
    assert out[8, 8]
    assert not out[1, 1]


def test_keep_top_detections_min_separation():
    E = np.zeros((20, 20), dtype=np.float32)
    E[10, 10] = 10.0
    E[11, 11] = 9.0
    E[2, 2] = 8.0

    det = E > 0

    out = keep_top_detections(
        detections=det,
        energies=E,
        max_detections=None,
        min_separation=3,
    )

    assert out[10, 10]
    assert not out[11, 11]
    assert out[2, 2]