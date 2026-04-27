import json

import numpy as np
import pytest

from src.synthetic import (
    MovingObject,
    generate_synthetic_sequence,
    make_single_disk,
    make_single_gaussian,
    make_two_objects,
    save_sequence,
    save_video_summary,
)


def test_generate_synthetic_sequence_default_shapes():
    video, masks, metadata = generate_synthetic_sequence(
        T=10,
        H=32,
        W=40,
        seed=0,
    )

    assert video.shape == (10, 32, 40)
    assert masks.shape == (1, 10, 32, 40)

    assert video.dtype == np.float32
    assert masks.dtype == np.float32

    assert metadata["T"] == 10
    assert metadata["H"] == 32
    assert metadata["W"] == 40
    assert len(metadata["objects"]) == 1
    assert len(metadata["velocities"]) == 1


def test_generate_single_gaussian_peak_location_at_t0():
    obj = MovingObject(
        kind="gaussian",
        center=(20.0, 12.0),  # (cx, cy)
        velocity=(0.0, 0.0),
        size=3.0,
        amplitude=1.0,
    )

    video, masks, metadata = generate_synthetic_sequence(
        T=3,
        H=32,
        W=40,
        objects=[obj],
        background=0.0,
        noise_std=0.0,
        normalize=False,
        clip=True,
        seed=0,
    )

    peak_y, peak_x = np.unravel_index(np.argmax(video[0]), video[0].shape)

    assert peak_x == 20
    assert peak_y == 12
    assert video[0, peak_y, peak_x] == pytest.approx(1.0)
    assert metadata["centers"] == [(20.0, 12.0)]
    assert metadata["velocities"] == [(0.0, 0.0)]


def test_gaussian_moves_right_when_vx_positive():
    obj = MovingObject(
        kind="gaussian",
        center=(10.0, 16.0),
        velocity=(2.0, 0.0),
        size=2.0,
        amplitude=1.0,
    )

    video, _, _ = generate_synthetic_sequence(
        T=4,
        H=32,
        W=40,
        objects=[obj],
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    peaks = [
        np.unravel_index(np.argmax(video[t]), video[t].shape)
        for t in range(4)
    ]

    # np returns (y, x)
    xs = [p[1] for p in peaks]
    ys = [p[0] for p in peaks]

    assert xs == [10, 12, 14, 16]
    assert ys == [16, 16, 16, 16]


def test_gaussian_moves_down_when_vy_positive():
    obj = MovingObject(
        kind="gaussian",
        center=(16.0, 10.0),
        velocity=(0.0, 2.0),
        size=2.0,
        amplitude=1.0,
    )

    video, _, _ = generate_synthetic_sequence(
        T=4,
        H=40,
        W=32,
        objects=[obj],
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    peaks = [
        np.unravel_index(np.argmax(video[t]), video[t].shape)
        for t in range(4)
    ]

    xs = [p[1] for p in peaks]
    ys = [p[0] for p in peaks]

    assert xs == [16, 16, 16, 16]
    assert ys == [10, 12, 14, 16]


def test_disk_mask_and_values_are_binary_like():
    obj = MovingObject(
        kind="disk",
        center=(16.0, 16.0),
        velocity=(0.0, 0.0),
        size=5.0,
        amplitude=0.8,
    )

    video, masks, _ = generate_synthetic_sequence(
        T=2,
        H=32,
        W=32,
        objects=[obj],
        background=0.0,
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    unique_video_values = np.unique(video)
    unique_mask_values = np.unique(masks)

    assert set(unique_mask_values.tolist()).issubset({0.0, 1.0})
    assert 0.8 in unique_video_values
    assert 1.0 in unique_mask_values


def test_square_mask_and_peak():
    obj = MovingObject(
        kind="square",
        center=(20.0, 15.0),
        velocity=(0.0, 0.0),
        size=4.0,
        amplitude=0.6,
    )

    video, masks, metadata = generate_synthetic_sequence(
        T=1,
        H=32,
        W=40,
        objects=[obj],
        background=0.0,
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    assert video.shape == (1, 32, 40)
    assert masks.shape == (1, 1, 32, 40)
    assert video[0, 15, 20] == pytest.approx(0.6)
    assert masks[0, 0, 15, 20] == pytest.approx(1.0)
    assert metadata["kinds"] == ["square"]


def test_unknown_object_kind_raises():
    obj = MovingObject(
        kind="triangle",
        center=(16.0, 16.0),
        velocity=(0.0, 0.0),
        size=4.0,
        amplitude=1.0,
    )

    with pytest.raises(ValueError, match="Unknown object kind"):
        generate_synthetic_sequence(
            T=1,
            H=32,
            W=32,
            objects=[obj],
        )


def test_multiple_objects_video_equals_sum_of_individual_videos():
    obj1 = MovingObject(
        kind="gaussian",
        center=(10.0, 16.0),
        velocity=(1.0, 0.0),
        size=2.0,
        amplitude=1.0,
    )

    obj2 = MovingObject(
        kind="disk",
        center=(24.0, 16.0),
        velocity=(-1.0, 0.0),
        size=3.0,
        amplitude=0.5,
    )

    video_both, masks_both, metadata = generate_synthetic_sequence(
        T=4,
        H=32,
        W=40,
        objects=[obj1, obj2],
        background=0.0,
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    video_1, _, _ = generate_synthetic_sequence(
        T=4,
        H=32,
        W=40,
        objects=[obj1],
        background=0.0,
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    video_2, _, _ = generate_synthetic_sequence(
        T=4,
        H=32,
        W=40,
        objects=[obj2],
        background=0.0,
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    assert video_both.shape == (4, 32, 40)
    assert masks_both.shape == (2, 4, 32, 40)
    assert metadata["velocities"] == [(1.0, 0.0), (-1.0, 0.0)]
    assert np.allclose(video_both, np.clip(video_1 + video_2, 0.0, 1.0))


def test_background_is_added():
    obj = MovingObject(
        kind="gaussian",
        center=(16.0, 16.0),
        velocity=(0.0, 0.0),
        size=2.0,
        amplitude=0.5,
    )

    video, _, metadata = generate_synthetic_sequence(
        T=1,
        H=32,
        W=32,
        objects=[obj],
        background=0.2,
        noise_std=0.0,
        normalize=False,
        clip=True,
    )

    assert metadata["background"] == 0.2
    assert video.min() >= 0.2 - 1e-6
    assert video.max() <= 0.7 + 1e-6


def test_noise_is_reproducible_with_seed():
    obj = MovingObject(
        kind="gaussian",
        center=(16.0, 16.0),
        velocity=(0.0, 0.0),
        size=2.0,
        amplitude=0.5,
    )

    video_a, _, _ = generate_synthetic_sequence(
        T=3,
        H=32,
        W=32,
        objects=[obj],
        noise_std=0.1,
        seed=123,
    )

    video_b, _, _ = generate_synthetic_sequence(
        T=3,
        H=32,
        W=32,
        objects=[obj],
        noise_std=0.1,
        seed=123,
    )

    assert np.allclose(video_a, video_b)


def test_normalize_maps_video_to_unit_interval():
    obj = MovingObject(
        kind="gaussian",
        center=(16.0, 16.0),
        velocity=(0.0, 0.0),
        size=2.0,
        amplitude=10.0,
    )

    video, _, metadata = generate_synthetic_sequence(
        T=1,
        H=32,
        W=32,
        objects=[obj],
        background=5.0,
        normalize=True,
        clip=True,
    )

    assert metadata["normalize"] is True
    assert video.min() == pytest.approx(0.0)
    assert video.max() == pytest.approx(1.0)


def test_clip_false_allows_values_above_one():
    obj = MovingObject(
        kind="gaussian",
        center=(16.0, 16.0),
        velocity=(0.0, 0.0),
        size=2.0,
        amplitude=2.0,
    )

    video, _, metadata = generate_synthetic_sequence(
        T=1,
        H=32,
        W=32,
        objects=[obj],
        background=0.0,
        normalize=False,
        clip=False,
    )

    assert metadata["clip"] is False
    assert video.max() > 1.0


def test_preset_single_gaussian():
    video, masks, metadata = make_single_gaussian()

    assert video.shape == (64, 128, 128)
    assert masks.shape == (1, 64, 128, 128)
    assert metadata["velocities"] == [(1.0, 0.0)]
    assert metadata["kinds"] == ["gaussian"]


def test_preset_single_disk():
    video, masks, metadata = make_single_disk()

    assert video.shape == (64, 128, 128)
    assert masks.shape == (1, 64, 128, 128)
    assert metadata["velocities"] == [(1.0, 0.5)]
    assert metadata["kinds"] == ["disk"]


def test_preset_two_objects():
    video, masks, metadata = make_two_objects()

    assert video.shape == (64, 128, 128)
    assert masks.shape == (2, 64, 128, 128)
    assert metadata["velocities"] == [(1.0, 0.0), (-0.5, -0.75)]
    assert metadata["kinds"] == ["gaussian", "disk"]


def test_save_sequence_writes_expected_files(tmp_path):
    video, masks, metadata = make_single_gaussian()

    save_sequence(
        video=video,
        masks=masks,
        metadata=metadata,
        out_dir=tmp_path,
        name="toy",
    )

    video_path = tmp_path / "toy.npy"
    masks_path = tmp_path / "toy_masks.npy"
    metadata_path = tmp_path / "toy_metadata.json"

    assert video_path.exists()
    assert masks_path.exists()
    assert metadata_path.exists()

    loaded_video = np.load(video_path)
    loaded_masks = np.load(masks_path)

    with open(metadata_path, "r") as f:
        loaded_metadata = json.load(f)

    assert loaded_video.shape == video.shape
    assert loaded_masks.shape == masks.shape
    assert loaded_metadata["T"] == metadata["T"]
    assert loaded_metadata["velocities"] == [[1.0, 0.0]]


def test_save_video_summary_writes_png(tmp_path):
    video, _, _ = make_single_gaussian()

    out_path = tmp_path / "summary.png"

    save_video_summary(video, out_path=out_path, title="Test Summary")

    assert out_path.exists()
    assert out_path.stat().st_size > 0