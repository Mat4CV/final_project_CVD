import numpy as np

from src.plane_scoring_detection.fft import (
    fft3_video,
    hann_window_3d,
    suppress_temporal_dc,
    subtract_temporal_baseline,
    preprocess_power_spectrum,
)


def test_fft3_video_shapes():
    video = np.random.default_rng(0).normal(size=(8, 16, 32)).astype(np.float32)

    F, ft, fy, fx = fft3_video(video, dt=1.0, shift=True)

    assert F.shape == video.shape
    assert ft.shape == (8,)
    assert fy.shape == (16,)
    assert fx.shape == (32,)


def test_fft3_video_frequency_axes_are_shifted():
    video = np.zeros((8, 16, 32), dtype=np.float32)

    _, ft, fy, fx = fft3_video(video, dt=1.0, shift=True)

    assert np.all(np.diff(ft) > 0)
    assert np.all(np.diff(fy) > 0)
    assert np.all(np.diff(fx) > 0)

    assert ft[0] < 0
    assert fy[0] < 0
    assert fx[0] < 0


def test_hann_window_3d_preserves_shape():
    video = np.ones((8, 16, 16), dtype=np.float32)

    out = hann_window_3d(video)

    assert out.shape == video.shape
    assert np.all(out >= 0)
    assert out.max() <= 1.0


def test_hann_window_3d_zeroes_boundaries():
    video = np.ones((8, 16, 16), dtype=np.float32)

    out = hann_window_3d(video)

    assert np.allclose(out[0], 0)
    assert np.allclose(out[-1], 0)
    assert np.allclose(out[:, 0, :], 0)
    assert np.allclose(out[:, -1, :], 0)
    assert np.allclose(out[:, :, 0], 0)
    assert np.allclose(out[:, :, -1], 0)


def test_suppress_temporal_dc_zeroes_center_band():
    power = np.ones((9, 8, 8), dtype=np.float32)
    ft = np.arange(-4, 5)

    out = suppress_temporal_dc(power, ft, dc_bins=1)

    idx0 = np.argmin(np.abs(ft))

    assert np.allclose(out[idx0 - 1 : idx0 + 2], 0)
    assert np.allclose(out[0], 1)
    assert np.allclose(out[-1], 1)


def test_subtract_temporal_baseline_nonnegative():
    power = np.ones((8, 4, 4), dtype=np.float32)
    power[3] = 10.0

    out = subtract_temporal_baseline(power, percentile=10.0)

    assert out.shape == power.shape
    assert np.all(out >= 0)


def test_preprocess_power_spectrum_returns_float32_and_nonnegative():
    video = np.random.default_rng(0).normal(size=(8, 16, 16)).astype(np.float32)

    F, ft, fy, fx = fft3_video(video)
    power = preprocess_power_spectrum(F, ft, dc_bins=1, baseline_percentile=10.0)

    assert power.shape == video.shape
    assert power.dtype == np.float32
    assert np.all(power >= 0)