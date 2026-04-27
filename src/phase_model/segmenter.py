from dataclasses import dataclass
from typing import List, Tuple
import warnings

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
    

class AdditiveImageSeparator:
    """
    Separate m additive component images from N composite frames.
 
    Each component image is assumed to translate at a distinct, known,
    constant velocity (pixels/frame).  Works for any m >= 2, and any N >= m
    (more frames gives better noise robustness).
 
    The book (Vernon 2001, Ch.3) states that for m components you need 2m
    equations (i.e. 2m frames) when velocities are UNKNOWN.  Here velocities
    are assumed already found (e.g. via Hough transform), so the unknowns
    are only the m phasors Fⁱ_t0 — making N >= m sufficient, with N >= 2m
    still recommended for conditioning.
 
    Parameters
    ----------
    velocities : list of (vx, vy) tuples, length m
        Known velocity of each component in pixels/frame.
    dt : float
        Time step between frames (default 1.0).
    eps : float
        Tikhonov regularisation for near-singular (degenerate) frequencies.
 
    Example
    -------
    sep = AdditiveImageSeparator(velocities=[(2, 1), (-1, 2), (0, -3)])
    imgs = sep.separate(frames)   # frames: (N, H, W) array
    img1, img2, img3 = imgs
    """
 
    def __init__(
        self,
        velocities: List[Tuple[float, float]],
        dt: float = 1.0,
        eps: float = 1e-8,
    ):
        self.velocities = [np.array(v, dtype=float) for v in velocities]
        self.m = len(velocities)
        self.dt = dt
        self.eps = eps
 
    # ── public ────────────────────────────────────────────────────────────────
 
    def separate(self, frames: np.ndarray) -> np.ndarray:
        """
        Recover all N frames for each of the m component images.
 
        Parameters
        ----------
        frames : (N, H, W) float array
            Composite image sequence.  frames[j] = sum_i f^i_tj.
 
        Returns
        -------
        sequences : (m, N, H, W) float array
            sequences[i, j] is component i at time t_j.
            sequences[i, 0] is the t=0 frame (same as before).
        """
        frames = np.asarray(frames, dtype=float)
        N, H, W = frames.shape
        m = self.m
 
        if N < m:
            raise ValueError(
                f"Need at least N={m} frames for m={m} components; got {N}. "
                f"Recommended N >= 2m = {2*m} for good conditioning."
            )
        if N < 2 * m:
            warnings.warn(
                f"N={N} < 2m={2*m}: fewer frames than the book recommends. "
                "Results may be inaccurate. Add more frames if possible.",
                stacklevel=2,
            )
 
        # ── 1. FFT every frame  →  shape (N, H, W) complex ────────────────
        F = np.stack([np.fft.fft2(frames[j]) for j in range(N)], axis=0)
 
        # ── 2. Angular spatial-frequency grids (FFT layout) ───────────────
        wx = 2 * np.pi * np.fft.fftfreq(W)   # (W,)
        wy = 2 * np.pi * np.fft.fftfreq(H)   # (H,)
        WX, WY = np.meshgrid(wx, wy)           # (H, W)
        HW = H * W
 
        # ── 3. Per-frame phase factors for each component ─────────────────
        #   ΔΦᵢ(wx,wy) = exp(-i (wx·vxᵢ + wy·vyᵢ) · dt)
        #   shape (HW,) per component
        dPhi = []
        for v in self.velocities:
            phi = np.exp(-1j * (WX * v[0] + WY * v[1]) * self.dt).ravel()
            dPhi.append(phi)                   # (HW,)
 
        # ── 4. Design matrix A  (Vandermonde in ΔΦ) ──────────────────────
        #   A[j, i, freq] = ΔΦᵢ(freq)^j
        #   We build shape (HW, N, m) — frequencies as the batch dimension.
        j_idx = np.arange(N)                   # (N,)
 
        # Each column: (HW,)^j broadcast → (N, HW), then transposed → (HW, N)
        cols = [(dPhi[i][np.newaxis, :] ** j_idx[:, np.newaxis]).T
                for i in range(m)]             # each (HW, N)
 
        A = np.stack(cols, axis=-1)            # (HW, N, m)
 
        # ── 5. Observation vector b  →  (HW, N, 1) ────────────────────────
        b = F.reshape(N, HW).T[:, :, np.newaxis]  # (HW, N, 1)
 
        # ── 6. Least-squares via normal equations: x = (AᴴA)⁻¹ Aᴴb ───────
        #   Solves for Fⁱ_t0 at every (wx,wy) simultaneously.
        #   AᴴA : (HW, m, m)
        #   Aᴴb : (HW, m, 1)
        AH = np.conj(A).transpose(0, 2, 1)    # (HW, m, N)
        AHA = AH @ A                           # (HW, m, m)
        AHb = AH @ b                           # (HW, m, 1)
 
        # Tikhonov regularisation: AᴴA + ε·I
        eye_m = np.eye(m, dtype=complex)[np.newaxis]  # (1, m, m)
        AHA += self.eps * eye_m
 
        x = np.linalg.solve(AHA, AHb)         # (HW, m, 1)
        x = x.squeeze(-1)                      # (HW, m)
 
        # x[:, i] = Fⁱ_t0(wx, wy)  — the t=0 Fourier spectrum of component i
 
        # ── 7. Propagate each component forward: Fⁱ_tj = Fⁱ_t0 · ΔΦᵢ^j ──
        #   then iFFT to get all N spatial frames per component.
        #
        #   sequences[i, j] = ifft2( Fⁱ_t0 · ΔΦᵢ^j )
        sequences = np.empty((m, N, H, W), dtype=float)
 
        for i in range(m):
            Fi_t0 = x[:, i].reshape(H, W)          # (H, W) complex
            dPhi_i = dPhi[i].reshape(H, W)          # (H, W) complex
 
            for j in range(N):
                # Apply j-th power of the phase ramp → shift component by j frames
                Fi_tj = Fi_t0 * (dPhi_i ** j)       # (H, W) complex
                sequences[i, j] = np.real(np.fft.ifft2(Fi_tj))
 
        return sequences
