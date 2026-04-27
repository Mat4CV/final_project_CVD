import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def small_frequency_grid():
    from src.phase_model.detection import frequency_grid

    H, W = 16, 20
    wx, wy = frequency_grid(H, W)
    return H, W, wx, wy


def nearest_velocity_distance(
    estimated: list[tuple[float, float]],
    target: tuple[float, float],
) -> float:
    """
    Distance from target velocity to the nearest estimated velocity.
    """
    target_arr = np.asarray(target, dtype=float)

    if len(estimated) == 0:
        return np.inf

    distances = [
        np.linalg.norm(np.asarray(v, dtype=float) - target_arr)
        for v in estimated
    ]

    return float(min(distances))


def match_velocity_sets(
    estimated: list[tuple[float, float]],
    expected: list[tuple[float, float]],
    tolerance: float,
) -> bool:
    """
    Checks whether every expected velocity has a nearby estimated velocity.
    Does not enforce one-to-one assignment.
    """
    return all(
        nearest_velocity_distance(estimated, target) <= tolerance
        for target in expected
    )


@pytest.fixture
def nearest_distance_fn():
    return nearest_velocity_distance


@pytest.fixture
def match_velocity_sets_fn():
    return match_velocity_sets