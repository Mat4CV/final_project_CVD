from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class VelocityGrid:
    """
    Velocity grid in physical units: pixels/frame.
    """
    v_min: float
    v_max: float
    num: int

    def values(self) -> np.ndarray:
        return np.linspace(self.v_min, self.v_max, self.num, dtype=np.float32)


@dataclass
class VelocityPlaneScorerConfig:
    sigma: float = 0.5
    keep_frac: float | None = 0.01
    max_keep: int = 5_000_000
    batch_size: int = 32
    use_soft_mask: bool = False

    # "auto", "numpy", or "torch"
    backend: str = "auto"

    # Used only when backend is torch or auto selects torch.
    device: str = "cuda"

    verbose: bool = True


def normalize_velocity(
    velocity_px_frame: np.ndarray,
    num_t: int,
    num_space: int,
) -> np.ndarray:
    """
    Convert px/frame velocity to normalized Fourier slope.

    v_norm = v_px_per_frame * num_t / num_space
    """
    return velocity_px_frame * (num_t / num_space)


class VelocityPlaneScorer:
    """
    Scores candidate 2D velocities by accumulating Fourier power near planes:

        ft + vx * fx + vy * fy = 0

    Public usage:

        scorer = VelocityPlaneScorer(velocity_grid, config)
        energies = scorer.compute(power, ft, fy, fx)

    Output convention:

        energies[i, j] corresponds to:
            vx = velocity_grid.values()[i]
            vy = velocity_grid.values()[j]
    """

    def __init__(
        self,
        velocity_grid: VelocityGrid,
        config: VelocityPlaneScorerConfig | None = None,
    ):
        self.velocity_grid = velocity_grid
        self.config = config or VelocityPlaneScorerConfig()

    def compute(
        self,
        power: np.ndarray,
        ft: np.ndarray,
        fy: np.ndarray,
        fx: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the velocity-plane energy image E(vx, vy).
        """
        self._validate_inputs(power, ft, fy, fx)

        backend = self._resolve_backend()

        if self.config.verbose:
            print("[VelocityPlaneScorer]")
            print(f"  backend: {backend}")
            print(f"  velocity grid: {self.velocity_grid.v_min} to {self.velocity_grid.v_max}")
            print(f"  num velocities: {self.velocity_grid.num}")
            print(f"  sigma: {self.config.sigma}")
            print(f"  keep_frac: {self.config.keep_frac}")

        flat = self._prepare_flat_volume(power, ft, fy, fx)

        if backend == "torch":
            return self._compute_torch(flat, power.shape)

        return self._compute_numpy(flat, power.shape)

    def _resolve_backend(self) -> str:
        backend = self.config.backend.lower()

        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError(
                f"Unknown backend '{self.config.backend}'. "
                "Use 'auto', 'numpy', or 'torch'."
            )

        if backend == "auto":
            return "torch" if torch.cuda.is_available() else "numpy"

        if backend == "torch":
            if self.config.device == "cuda" and not torch.cuda.is_available():
                if self.config.verbose:
                    print("  CUDA requested but unavailable. Falling back to NumPy.")
                return "numpy"
            return "torch"

        return "numpy"

    def _validate_inputs(
        self,
        power: np.ndarray,
        ft: np.ndarray,
        fy: np.ndarray,
        fx: np.ndarray,
    ) -> None:
        if power.ndim != 3:
            raise ValueError(f"power must have shape (T,H,W), got {power.shape}")

        T, H, W = power.shape

        if ft.shape[0] != T:
            raise ValueError(f"ft has length {len(ft)}, expected {T}")
        if fy.shape[0] != H:
            raise ValueError(f"fy has length {len(fy)}, expected {H}")
        if fx.shape[0] != W:
            raise ValueError(f"fx has length {len(fx)}, expected {W}")

        if self.velocity_grid.num <= 0:
            raise ValueError("velocity_grid.num must be positive")

        if self.config.sigma <= 0:
            raise ValueError("sigma must be positive")

    def _prepare_flat_volume(
        self,
        power: np.ndarray,
        ft: np.ndarray,
        fy: np.ndarray,
        fx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Flatten the frequency volume and prune weak coefficients.

        Returns a dictionary containing:
            p, ft, fy, fx
        """
        FT, FY, FX = np.meshgrid(ft, fy, fx, indexing="ij")

        p = power.reshape(-1).astype(np.float32)
        ft_f = FT.reshape(-1).astype(np.float32)
        fy_f = FY.reshape(-1).astype(np.float32)
        fx_f = FX.reshape(-1).astype(np.float32)

        p, ft_f, fy_f, fx_f = self._topk_prune(p, ft_f, fy_f, fx_f)

        return {
            "p": p,
            "ft": ft_f,
            "fy": fy_f,
            "fx": fx_f,
        }

    def _topk_prune(
        self,
        p: np.ndarray,
        ft: np.ndarray,
        fy: np.ndarray,
        fx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Keep the strongest Fourier coefficients.
        """
        keep_frac = self.config.keep_frac

        if keep_frac is None:
            return p, ft, fy, fx

        n = p.size
        k = int(n * float(keep_frac))
        k = max(1, min(k, int(self.config.max_keep), n))

        idx = np.argpartition(p, -k)[-k:]

        if self.config.verbose:
            print(f"  kept coefficients: {k} / {n}")

        return p[idx], ft[idx], fy[idx], fx[idx]

    def _normalized_velocity_axes(
        self,
        shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalized vx and vy grids.
        """
        T, H, W = shape
        v_px = self.velocity_grid.values()

        vx = normalize_velocity(v_px, T, W).astype(np.float32)
        vy = normalize_velocity(v_px, T, H).astype(np.float32)

        return vx, vy

    def _compute_numpy(
        self,
        flat: dict[str, np.ndarray],
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        """
        NumPy backend.
        """
        p = flat["p"]
        ft = flat["ft"]
        fy = flat["fy"]
        fx = flat["fx"]

        vx_axis, vy_axis = self._normalized_velocity_axes(shape)
        Nv = self.velocity_grid.num

        energies = np.zeros((Nv, Nv), dtype=np.float32)

        for i, vx in tqdm(
            enumerate(vx_axis),
            total=Nv,
            desc="Velocity plane energy",
            disable=not self.config.verbose,
        ):
            base = ft + vx * fx

            # Vectorized over all vy at once.
            residual = base[:, None] + fy[:, None] * vy_axis[None, :]

            if self.config.use_soft_mask:
                weights = np.exp(-(residual**2) / (2.0 * self.config.sigma**2))
                energies[i, :] = np.sum(p[:, None] * weights, axis=0)
            else:
                mask = np.abs(residual) < self.config.sigma
                energies[i, :] = np.sum(p[:, None] * mask, axis=0)

        return energies

    def _compute_torch(
        self,
        flat: dict[str, np.ndarray],
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Torch backend. Works on CUDA or CPU.
        """
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        p = torch.as_tensor(flat["p"], device=device, dtype=torch.float32)
        ft = torch.as_tensor(flat["ft"], device=device, dtype=torch.float32)
        fy = torch.as_tensor(flat["fy"], device=device, dtype=torch.float32)
        fx = torch.as_tensor(flat["fx"], device=device, dtype=torch.float32)

        vx_axis, vy_axis = self._normalized_velocity_axes(shape)

        vx = torch.as_tensor(vx_axis, device=device, dtype=torch.float32)
        vy = torch.as_tensor(vy_axis, device=device, dtype=torch.float32)

        Nv = self.velocity_grid.num
        energies = torch.zeros((Nv, Nv), device=device, dtype=torch.float32)

        sigma = float(self.config.sigma)
        batch_size = int(self.config.batch_size)

        for i in tqdm(
            range(Nv),
            desc="Velocity plane energy",
            disable=not self.config.verbose,
        ):
            base = ft + vx[i] * fx
            base = base[:, None]

            for j0 in range(0, Nv, batch_size):
                j1 = min(j0 + batch_size, Nv)
                vy_batch = vy[j0:j1]

                residual = base + fy[:, None] * vy_batch[None, :]

                if self.config.use_soft_mask:
                    weights = torch.exp(-(residual**2) / (2.0 * sigma**2))
                    batch_energy = (p[:, None] * weights).sum(dim=0)
                else:
                    mask = torch.abs(residual) < sigma
                    batch_energy = (p[:, None] * mask).sum(dim=0)

                energies[i, j0:j1] = batch_energy

        return energies.detach().cpu().numpy()