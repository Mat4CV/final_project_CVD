from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cfar import CFARResult, keep_top_detections, rank_cfar_2d
from .energy import (
    VelocityGrid,
    VelocityPlaneScorer,
    VelocityPlaneScorerConfig,
)
from .fft import fft3_video, hann_window_3d, preprocess_power_spectrum


@dataclass
class FourierMotionConfig:
    velocity_bounds: tuple[float, float, int]

    dt: float = 1.0

    sigma: float = 0.5
    alpha: float = 1e-3

    dc_bins: int = 1
    baseline_percentile: float = 10.0

    cfar_window: int | None = None
    cfar_guard: int = 1

    use_hann_window: bool = True
    use_gpu: bool = True

    keep_frac: float | None = 0.01
    max_keep: int = 5_000_000
    batch_size: int = 32

    max_detections: int | None = None
    min_detection_separation: int = 0

    use_soft_mask: bool = False
    verbose: bool = True


@dataclass
class FourierMotionResult:
    velocity_grid: np.ndarray
    energies: np.ndarray
    detections: np.ndarray
    detected_velocities: np.ndarray
    detected_energies: np.ndarray

    cfar: CFARResult

    F: np.ndarray
    ft: np.ndarray
    fy: np.ndarray
    fx: np.ndarray

    config: FourierMotionConfig


class FourierMotionDetector:
    """
    Main object for Fourier-domain velocity-plane detection.

    Pipeline:
        video
          -> optional Hann window
          -> 3D FFT
          -> power spectrum preprocessing
          -> E(vx, vy)
          -> 2D rank CFAR
          -> detected velocities
    """

    def __init__(self, config: FourierMotionConfig):
        self.config = config

    @classmethod
    def from_bounds(
        cls,
        velocity_bounds: tuple[float, float, int],
        **kwargs,
    ) -> "FourierMotionDetector":
        config = FourierMotionConfig(
            velocity_bounds=velocity_bounds,
            **kwargs,
        )
        return cls(config)

    def detect(self, video: np.ndarray) -> FourierMotionResult:
        cfg = self.config

        video = np.asarray(video, dtype=np.float32)

        if video.ndim != 3:
            raise ValueError(f"video must have shape (T,H,W), got {video.shape}")

        if cfg.verbose:
            print("[FourierMotionDetector]")
            print(f"  video shape: {video.shape}")
            print(f"  velocity bounds: {cfg.velocity_bounds}")

        if cfg.use_hann_window:
            video_proc = hann_window_3d(video)
        else:
            video_proc = video

        F, ft, fy, fx = fft3_video(
            video_proc,
            dt=cfg.dt,
            shift=True,
        )

        power = preprocess_power_spectrum(
            F,
            ft=ft,
            dc_bins=cfg.dc_bins,
            baseline_percentile=cfg.baseline_percentile,
        )

        v_min, v_max, num_v = cfg.velocity_bounds

        velocity_grid = VelocityGrid(
            v_min=v_min,
            v_max=v_max,
            num=num_v,
        )

        scorer_config = VelocityPlaneScorerConfig(
            sigma=cfg.sigma,
            keep_frac=cfg.keep_frac,
            max_keep=cfg.max_keep,
            batch_size=cfg.batch_size,
            use_soft_mask=cfg.use_soft_mask,
            backend="auto" if cfg.use_gpu else "numpy",
            device="cuda",
            verbose=cfg.verbose,
        )

        scorer = VelocityPlaneScorer(
            velocity_grid=velocity_grid,
            config=scorer_config,
        )

        energies = scorer.compute(
            power=power,
            ft=ft,
            fy=fy,
            fx=fx,
        )

        cfar_window = cfg.cfar_window
        if cfar_window is None:
            cfar_window = max(2, num_v // 8)

        cfar = rank_cfar_2d(
            energies,
            alpha=cfg.alpha,
            window=cfar_window,
            guard=cfg.cfar_guard,
        )

        detections = keep_top_detections(
            cfar.detections,
            energies,
            max_detections=cfg.max_detections,
            min_separation=cfg.min_detection_separation,
        )

        detected_velocities, detected_energies = self._detections_to_velocities(
            detections=detections,
            energies=energies,
            velocity_grid=velocity_grid.values(),
        )

        if cfg.verbose:
            print(f"  detections: {len(detected_velocities)}")
            for velocity, energy in zip(detected_velocities[:10], detected_energies[:10]):
                vx, vy = velocity
                print(f"    vx={vx: .3f}, vy={vy: .3f}, energy={energy:.3e}")

        return FourierMotionResult(
            velocity_grid=velocity_grid.values(),
            energies=energies,
            detections=detections,
            detected_velocities=detected_velocities,
            detected_energies=detected_energies,
            cfar=cfar,
            F=F,
            ft=ft,
            fy=fy,
            fx=fx,
            config=cfg,
        )

    @staticmethod
    def _detections_to_velocities(
        detections: np.ndarray,
        energies: np.ndarray,
        velocity_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argwhere(detections)

        if idx.size == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        velocities = []
        det_energies = []

        for i, j in idx:
            vx = velocity_grid[i]
            vy = velocity_grid[j]
            velocities.append((vx, vy))
            det_energies.append(energies[i, j])

        velocities = np.asarray(velocities, dtype=np.float32)
        det_energies = np.asarray(det_energies, dtype=np.float32)

        order = np.argsort(det_energies)[::-1]

        return velocities[order], det_energies[order]