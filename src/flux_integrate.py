"""
flux_integrator.py
==================
Evaluate a continuous flux function flux(x, y, t) on a discrete W×H×T grid
by integrating over each voxel's spatial and temporal extents.

Each voxel (i, j, k) covers:
    x ∈ [i·dx, (i+1)·dx]       where dx = W / grid_W
    y ∈ [j·dy, (j+1)·dy]       where dy = H / grid_H
    t ∈ [k·dt, (k+1)·dt]       where dt = tau / grid_T

The intensity at voxel (i, j, k) is:
    I[i, j, k] = ∫∫∫ flux(x, y, t) dx dy dt
               ≈ numerical triple integral over the voxel domain
"""

import numpy as np
from scipy.integrate import dblquad, tplquad
from typing import Callable
import warnings
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core integrators
# ---------------------------------------------------------------------------

def integrate_voxel_grid(
    flux: Callable[[float, float, float], float],
    W: float,
    H: float,
    tau: float,
    grid_W: int,
    grid_H: int,
    grid_T: int,
    method: str = "quadrature",
    n_samples: int = 8,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Integrate flux(x, y, t) over every voxel on a grid_W × grid_H × grid_T grid.

    Parameters
    ----------
    flux     : callable(x, y, t) → float — continuous flux function.
    W, H     : spatial domain extents (x ∈ [0,W], y ∈ [0,H]).
    tau      : temporal domain extent (t ∈ [0,tau]).
    grid_W   : number of spatial columns (x-axis bins).
    grid_H   : number of spatial rows    (y-axis bins).
    grid_T   : number of temporal frames (t-axis bins).
    method   : 'quadrature' — adaptive scipy quadrature (accurate, slower)
               'midpoint'   — midpoint/Newton-Cotes rule (fast approximation)
               'montecarlo' — Monte Carlo integration  (for high dimensions)
    n_samples: points per dimension for 'midpoint' or 'montecarlo'.
    tol      : relative/absolute tolerance passed to quadrature.

    Returns
    -------
    intensities : np.ndarray of shape (grid_T, grid_H, grid_W)
        I[k, j, i] = ∫∫∫_{voxel(i,j,k)} flux(x,y,t) dx dy dt
    """
    dx = W   / grid_W
    dy = H   / grid_H
    dt = tau / grid_T

    intensities = np.zeros((grid_T, grid_H, grid_W))

    if method == "quadrature":
        _fill_quadrature(flux, intensities, dx, dy, dt, grid_W, grid_H, grid_T, tol)
    elif method == "midpoint":
        _fill_midpoint(flux, intensities, dx, dy, dt, grid_W, grid_H, grid_T, n_samples)
    elif method == "montecarlo":
        _fill_montecarlo(flux, intensities, dx, dy, dt, grid_W, grid_H, grid_T, n_samples)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'quadrature', 'midpoint', or 'montecarlo'.")

    return intensities


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def _fill_quadrature(flux, out, dx, dy, dt, grid_W, grid_H, grid_T, tol):
    """Adaptive Gauss-Kronrod quadrature via scipy.integrate.tplquad."""
    total = grid_W * grid_H * grid_T
    done  = 0
    tqdm_iter = tqdm(total=total)
    for k in range(grid_T):
        t0, t1 = k * dt, (k + 1) * dt
        for j in range(grid_H):
            y0, y1 = j * dy, (j + 1) * dy
            for i in range(grid_W):
                x0, x1 = i * dx, (i + 1) * dx
                val, err = tplquad(
                    flux,
                    t0, t1,                         # outermost: t
                    y0, y1,                         # middle:    y
                    x0, x1,                         # innermost: x
                    epsabs=tol, epsrel=tol,
                )
                out[k, j, i] = val
                done += 1
                if done % max(1, total // 20) == 0:
                    print(f"  quadrature progress: {100*done/total:.0f}%")
                tqdm_iter.update(1)
    tqdm_iter.close()


def _fill_midpoint(flux, out, dx, dy, dt, grid_W, grid_H, grid_T, n):
    """
    Newton-Cotes / composite midpoint rule.
    Samples the flux on an n×n×n sub-grid of each voxel and sums.
    """
    # sub-cell offsets (midpoints of n equal sub-intervals)
    offsets = (np.arange(n) + 0.5) / n          # shape (n,)
    sub_vol = (dx * dy * dt) / (n ** 3)
    tqdm_iter = tqdm(total=grid_W * grid_H * grid_T)

    for k in range(grid_T):
        t0 = k * dt
        ts = t0 + offsets * dt                   # shape (n,)
        for j in range(grid_H):
            y0 = j * dy
            ys = y0 + offsets * dy               # shape (n,)
            for i in range(grid_W):
                x0 = i * dx
                xs = x0 + offsets * dx           # shape (n,)
                # vectorised evaluation over the n³ sub-samples
                X, Y, T_ = np.meshgrid(xs, ys, ts, indexing='ij')
                vals = np.vectorize(flux)(X, Y, T_)
                out[k, j, i] = vals.sum() * sub_vol
                tqdm_iter.update(1)
    tqdm_iter.close()


def _fill_montecarlo(flux, out, dx, dy, dt, grid_W, grid_H, grid_T, n):
    """
    Monte Carlo integration using n³ random samples per voxel.
    Variance ∝ 1/n³; useful when flux is expensive or high-dimensional.
    """
    rng    = np.random.default_rng(seed=0)
    vol    = dx * dy * dt
    n_pts  = n ** 3

    for k in range(grid_T):
        t0 = k * dt
        for j in range(grid_H):
            y0 = j * dy
            for i in range(grid_W):
                x0 = i * dx
                xs = x0 + rng.random(n_pts) * dx
                ys = y0 + rng.random(n_pts) * dy
                ts = t0 + rng.random(n_pts) * dt
                vals = np.vectorize(flux)(xs, ys, ts)
                out[k, j, i] = vals.mean() * vol


# ---------------------------------------------------------------------------
# Convenience: vectorised flux support
# ---------------------------------------------------------------------------

def vectorised_midpoint(
    flux_vec: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    W: float, H: float, tau: float,
    grid_W: int, grid_H: int, grid_T: int,
    n: int = 8,
) -> np.ndarray:
    """
    Fast midpoint rule when flux accepts vectorised numpy arrays.

    flux_vec(X, Y, T) must accept 3-D broadcastable arrays and return the
    same shape — avoids Python-level loops over individual sample points.

    Returns
    -------
    np.ndarray of shape (grid_T, grid_H, grid_W)
    """
    dx = W   / grid_W
    dy = H   / grid_H
    dt = tau / grid_T
    sub_vol = (dx * dy * dt) / (n ** 3)

    offsets = (np.arange(n) + 0.5) / n   # (n,)

    # Build coordinate arrays for ALL voxel centres simultaneously
    # shape of each: (grid_T, grid_H, grid_W, n, n, n)
    i_idx = np.arange(grid_W)
    j_idx = np.arange(grid_H)
    k_idx = np.arange(grid_T)

    # voxel-origin arrays: (grid_T,1,1,1,1,1) etc. → broadcast-friendly
    x0 = (i_idx * dx).reshape(1, 1, grid_W, 1, 1, 1)
    y0 = (j_idx * dy).reshape(1, grid_H, 1, 1, 1, 1)
    t0 = (k_idx * dt).reshape(grid_T, 1, 1, 1, 1, 1)

    ox = (offsets * dx).reshape(1, 1, 1, n, 1, 1)
    oy = (offsets * dy).reshape(1, 1, 1, 1, n, 1)
    ot = (offsets * dt).reshape(1, 1, 1, 1, 1, n)

    X = x0 + ox    # (grid_T, grid_H, grid_W, n, n, n)
    Y = y0 + oy
    T_ = t0 + ot

    vals = flux_vec(X, Y, T_)            # same shape
    intensities = vals.sum(axis=(-3, -2, -1)) * sub_vol   # (grid_T, grid_H, grid_W)
    return intensities
