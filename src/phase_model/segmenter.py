from dataclasses import dataclass

import numpy as np

from .detection import (
    frequency_grid,
    phase_hough_votes,
    reconstruct_from_spectra,
    solve_phasors_prony,
    solve_two_phasors_closed_form,
    sort_components_by_velocity,
    top_k_peaks,
)


@dataclass
class FourierVisionConfig:
    """
    Configuration for the Fourier Vision Chapter 3 additive model.

    Args:
        velocity_bounds:
            Search interval for both vx and vy.
        velocity_bins:
            Number of grid samples per velocity axis.
        num_components:
            Number of additive components assumed in the phasor model.
            For the Prony solver, the video must contain at least 2 * num_components frames.
        num_velocities:
            Number of velocity peaks to return from the Hough accumulator.
        dt:
            Time step between consecutive frames.
        eps:
            Numerical threshold.
        min_frequency_radius:
            Frequencies with sqrt(wx^2 + wy^2) below this value are ignored.
        hough_sigma:
            Soft angular tolerance used by Hough voting.
        peak_min_separation:
            Non-maximum suppression radius in velocity-grid cells.
        use_magnitude_weights:
            If True, weight Hough votes by recovered Fourier magnitude.
        solver:
            Either "prony" or "closed_form_m2".
    """

    velocity_bounds: tuple[float, float] = (-5.0, 5.0)
    velocity_bins: int = 201

    num_components: int = 2
    num_velocities: int = 2

    dt: float = 1.0
    eps: float = 1e-8
    min_frequency_radius: float = 0.2
    hough_sigma: float = 0.15
    peak_min_separation: int = 5
    use_magnitude_weights: bool = True

    solver: str = "prony"


@dataclass
class FourierVisionResult:
    """
    Result returned by FourierVisionSegmenter.

    Attributes:
        velocities:
            Estimated velocities as [(vx, vy), ...].
        hough:
            Velocity-space Hough accumulator.
        vx_values:
            Horizontal velocity grid.
        vy_values:
            Vertical velocity grid.
        components:
            Reconstructed spatial-domain component images, if segment() was used.
        spectra:
            Recovered object-specific Fourier spectra, if segment() was used.
        valid:
            Valid-frequency mask.
        z:
            Estimated phase-step fields with shape m x H x W.
        A:
            Estimated Fourier amplitudes with shape m x H x W.
    """

    velocities: list[tuple[float, float]]
    hough: np.ndarray
    vx_values: np.ndarray
    vy_values: np.ndarray

    components: list[np.ndarray] | None = None
    spectra: list[np.ndarray] | None = None
    valid: np.ndarray | None = None

    z: np.ndarray | None = None
    A: np.ndarray | None = None


class FourierVisionSegmenter:
    """
    Compact implementation of the Fourier Vision Chapter 3 additive model.

    The assumed model is:

        F_t(wx, wy) = sum_i A_i(wx, wy) z_i(wx, wy)^t

    where each phase step z_i corresponds to a translating component:

        z_i = exp(-j (wx vx_i + wy vy_i) dt)

    Velocity detection supports arbitrary num_components and num_velocities.
    Full component reconstruction is also implemented generically, but it is
    most stable for m=2 and clean synthetic m=3/m=4 additive sequences.
    """

    def __init__(self, config: FourierVisionConfig | None = None):
        self.config = config or FourierVisionConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.velocity_bins < 2:
            raise ValueError("velocity_bins must be at least 2.")

        if cfg.num_components < 1:
            raise ValueError("num_components must be at least 1.")

        if cfg.num_velocities < 1:
            raise ValueError("num_velocities must be at least 1.")

        if cfg.dt <= 0:
            raise ValueError("dt must be positive.")

        if cfg.eps <= 0:
            raise ValueError("eps must be positive.")

        if cfg.hough_sigma <= 0:
            raise ValueError("hough_sigma must be positive.")

        if cfg.peak_min_separation < 0:
            raise ValueError("peak_min_separation must be nonnegative.")

        if cfg.solver not in {"prony", "closed_form_m2"}:
            raise ValueError('solver must be either "prony" or "closed_form_m2".')

        if cfg.solver == "closed_form_m2" and cfg.num_components != 2:
            raise ValueError('solver="closed_form_m2" requires num_components=2.')

    def _fourier_sequence(self, video: np.ndarray) -> np.ndarray:
        """
        Convert a T x H x W video into a T x H x W sequence of 2D FFTs.
        """
        if video.ndim != 3:
            raise ValueError("video must have shape T x H x W.")

        video = video.astype(np.float64, copy=False)

        return np.stack(
            [np.fft.fft2(video[t]) for t in range(video.shape[0])],
            axis=0,
        )

    def _solve_phasors(
        self,
        F_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate phase steps and amplitudes from a Fourier sequence.
        """
        cfg = self.config

        if cfg.solver == "closed_form_m2":
            return solve_two_phasors_closed_form(F_seq, eps=cfg.eps)

        return solve_phasors_prony(
            F_seq,
            num_components=cfg.num_components,
            eps=cfg.eps,
        )

    def _prepare(
        self,
        video: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Shared preprocessing for detect() and segment().

        Returns:
            wx, wy:
                Frequency grids.
            z:
                Phase steps with shape m x H x W.
            A:
                Fourier amplitudes with shape m x H x W.
            valid:
                Valid-frequency mask.
            vx_values, vy_values:
                Velocity grids.
        """
        cfg = self.config

        if video.ndim != 3:
            raise ValueError("video must have shape T x H x W.")

        T, H, W = video.shape

        if cfg.solver == "closed_form_m2":
            min_frames = 4
        else:
            min_frames = 2 * cfg.num_components

        if T < min_frames:
            raise ValueError(
                f"Not enough frames for solver={cfg.solver}. "
                f"Need at least {min_frames}, got {T}."
            )

        F_seq = self._fourier_sequence(video)
        z, A, valid = self._solve_phasors(F_seq)

        wx, wy = frequency_grid(H, W)

        radius = np.sqrt(wx**2 + wy**2)
        valid &= radius >= cfg.min_frequency_radius

        vx_min, vx_max = cfg.velocity_bounds

        if vx_min >= vx_max:
            raise ValueError("velocity_bounds must satisfy min < max.")

        vx_values = np.linspace(vx_min, vx_max, cfg.velocity_bins)
        vy_values = np.linspace(vx_min, vx_max, cfg.velocity_bins)

        return wx, wy, z, A, valid, vx_values, vy_values

    def _compute_hough(
        self,
        z: np.ndarray,
        A: np.ndarray,
        wx: np.ndarray,
        wy: np.ndarray,
        valid: np.ndarray,
        vx_values: np.ndarray,
        vy_values: np.ndarray,
    ) -> np.ndarray:
        """
        Build velocity-space Hough accumulator from phase-step fields.
        """
        cfg = self.config

        weights = np.abs(A) if cfg.use_magnitude_weights else None

        return phase_hough_votes(
            z_fields=z,
            wx=wx,
            wy=wy,
            valid=valid,
            vx_values=vx_values,
            vy_values=vy_values,
            dt=cfg.dt,
            weights=weights,
            sigma=cfg.hough_sigma,
        )

    def detect(self, video: np.ndarray) -> FourierVisionResult:
        """
        Estimate the top-K velocities from the Fourier Vision phase model.

        This method does not reconstruct component images.
        """
        cfg = self.config

        wx, wy, z, A, valid, vx_values, vy_values = self._prepare(video)

        hough = self._compute_hough(
            z=z,
            A=A,
            wx=wx,
            wy=wy,
            valid=valid,
            vx_values=vx_values,
            vy_values=vy_values,
        )

        velocities = top_k_peaks(
            accumulator=hough,
            vx_values=vx_values,
            vy_values=vy_values,
            k=cfg.num_velocities,
            min_separation=cfg.peak_min_separation,
        )

        return FourierVisionResult(
            velocities=velocities,
            hough=hough,
            vx_values=vx_values,
            vy_values=vy_values,
            valid=valid,
            z=z,
            A=A,
        )

    def segment(self, video: np.ndarray) -> FourierVisionResult:
        """
        Estimate velocities and reconstruct additive components.

        For reconstruction, the number of detected velocities must match
        the number of assumed phasor components.
        """
        cfg = self.config

        if cfg.num_velocities != cfg.num_components:
            raise ValueError(
                "For component reconstruction, num_velocities must equal "
                "num_components. Use detect(video) if you only want velocity peaks."
            )

        wx, wy, z, A, valid, vx_values, vy_values = self._prepare(video)

        hough = self._compute_hough(
            z=z,
            A=A,
            wx=wx,
            wy=wy,
            valid=valid,
            vx_values=vx_values,
            vy_values=vy_values,
        )

        velocities = top_k_peaks(
            accumulator=hough,
            vx_values=vx_values,
            vy_values=vy_values,
            k=cfg.num_velocities,
            min_separation=cfg.peak_min_separation,
        )

        if len(velocities) != cfg.num_components:
            raise RuntimeError(
                f"Expected {cfg.num_components} velocities, "
                f"but peak selection returned {len(velocities)}."
            )

        spectra = sort_components_by_velocity(
            A=A,
            z=z,
            velocities=velocities,
            wx=wx,
            wy=wy,
            valid=valid,
            dt=cfg.dt,
        )

        components = reconstruct_from_spectra(spectra)

        return FourierVisionResult(
            velocities=velocities,
            hough=hough,
            vx_values=vx_values,
            vy_values=vy_values,
            components=components,
            spectra=spectra,
            valid=valid,
            z=z,
            A=A,
        )