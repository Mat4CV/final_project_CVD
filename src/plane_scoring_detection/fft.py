from __future__ import annotations

import numpy as np


def fft3_video(
    video: np.ndarray,
    dt: float = 1.0,
    shift: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 3D FFT of a video.

    Input convention:
        video[t, y, x]

    Output:
        F[ft, fy, fx]
    """
    if video.ndim != 3:
        raise ValueError(f"video must have shape (T,H,W), got {video.shape}")

    T, H, W = video.shape

    F = np.fft.fftn(video, axes=(0, 1, 2))

    # Dimensionless frequency-bin coordinates.
    # This makes the plane relation compatible with px/frame velocities.
    ft = np.fft.fftfreq(T, d=dt) * dt * T
    fy = np.fft.fftfreq(H, d=1.0) * H
    fx = np.fft.fftfreq(W, d=1.0) * W

    if shift:
        F = np.fft.fftshift(F)
        ft = np.fft.fftshift(ft)
        fy = np.fft.fftshift(fy)
        fx = np.fft.fftshift(fx)

    return F, ft, fy, fx


def hann_window_3d(video: np.ndarray) -> np.ndarray:
    """
    Apply a separable 3D Hann window to reduce spectral leakage.
    """
    if video.ndim != 3:
        raise ValueError(f"video must have shape (T,H,W), got {video.shape}")

    T, H, W = video.shape

    wt = np.hanning(T)[:, None, None]
    wy = np.hanning(H)[None, :, None]
    wx = np.hanning(W)[None, None, :]

    return video * wt * wy * wx


def suppress_temporal_dc(
    power: np.ndarray,
    ft: np.ndarray,
    dc_bins: int = 1,
) -> np.ndarray:
    """
    Suppress temporal frequencies near ft = 0.

    power has shape:
        (T, H, W)
    """
    if dc_bins <= 0:
        return power

    out = power.copy()

    idx0 = int(np.argmin(np.abs(ft)))
    lo = max(0, idx0 - dc_bins)
    hi = min(power.shape[0], idx0 + dc_bins + 1)

    out[lo:hi, :, :] = 0.0
    return out


def subtract_temporal_baseline(
    power: np.ndarray,
    percentile: float = 10.0,
) -> np.ndarray:
    """
    Subtract a per-spatial-frequency baseline estimated across time.

    This is applied independently for each (fy, fx) column.
    """
    baseline = np.percentile(power, percentile, axis=0, keepdims=True)
    out = power - baseline
    out[out < 0] = 0.0
    return out


def preprocess_power_spectrum(
    F: np.ndarray,
    ft: np.ndarray,
    dc_bins: int = 1,
    baseline_percentile: float = 10.0,
) -> np.ndarray:
    """
    Convert Fourier volume to cleaned power spectrum.
    """
    power = np.abs(F) ** 2
    power = suppress_temporal_dc(power, ft=ft, dc_bins=dc_bins)
    power = subtract_temporal_baseline(power, percentile=baseline_percentile)
    return power.astype(np.float32)