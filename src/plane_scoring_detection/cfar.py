from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CFARResult:
    statistic: np.ndarray
    background: np.ndarray
    detections: np.ndarray
    thresholds: np.ndarray


def _rank_threshold(alpha: float, n_train: int) -> int:
    """
    Distribution-free rank threshold.

    Under H0, the rank of the CUT among N training cells plus itself
    is uniform on {0, ..., N}.
    """
    k = int(np.floor(alpha * (n_train + 1)))
    k = max(1, min(k, n_train + 1))
    return n_train - k + 1


def rank_cfar_2d(
    energies: np.ndarray,
    alpha: float = 1e-3,
    window: int = 8,
    guard: int = 1,
) -> CFARResult:
    """
    2D rank-based nonparametric CFAR.

    For each cell under test, compare its energy to neighboring training cells.
    """
    E = np.asarray(energies, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"energies must be 2D, got shape {E.shape}")

    if window <= guard:
        raise ValueError("window must be larger than guard")

    H, W = E.shape

    statistic = np.zeros_like(E, dtype=float)
    background = np.zeros_like(E, dtype=float)
    thresholds = np.zeros_like(E, dtype=float)
    detections = np.zeros_like(E, dtype=bool)

    for r in range(H):
        r0 = max(0, r - window)
        r1 = min(H, r + window + 1)

        gr0 = max(0, r - guard)
        gr1 = min(H, r + guard + 1)

        for c in range(W):
            c0 = max(0, c - window)
            c1 = min(W, c + window + 1)

            gc0 = max(0, c - guard)
            gc1 = min(W, c + guard + 1)

            patch = E[r0:r1, c0:c1]
            train_mask = np.ones_like(patch, dtype=bool)

            pr0 = gr0 - r0
            pr1 = gr1 - r0
            pc0 = gc0 - c0
            pc1 = gc1 - c0

            train_mask[pr0:pr1, pc0:pc1] = False

            training = patch[train_mask]

            if training.size == 0:
                training = E.ravel()

            n_train = training.size

            rank = np.sum(E[r, c] > training)
            threshold = _rank_threshold(alpha, n_train)

            statistic[r, c] = rank
            background[r, c] = np.median(training)
            thresholds[r, c] = threshold
            detections[r, c] = rank >= threshold

    return CFARResult(
        statistic=statistic,
        background=background,
        detections=detections,
        thresholds=thresholds,
    )


def keep_top_detections(
    detections: np.ndarray,
    energies: np.ndarray,
    max_detections: int | None = None,
    min_separation: int = 0,
) -> np.ndarray:
    """
    Post-process a detection mask.

    Keeps strongest detections, optionally enforcing a minimum index-space separation.
    """
    det_idx = np.argwhere(detections)

    if det_idx.size == 0:
        return detections

    det_energy = energies[detections]
    order = np.argsort(det_energy)[::-1]
    det_idx = det_idx[order]

    selected = []

    for idx in det_idx:
        if max_detections is not None and len(selected) >= max_detections:
            break

        if min_separation > 0:
            too_close = False
            for prev in selected:
                if np.linalg.norm(idx - prev, ord=np.inf) < min_separation:
                    too_close = True
                    break

            if too_close:
                continue

        selected.append(idx)

    out = np.zeros_like(detections, dtype=bool)

    for r, c in selected:
        out[r, c] = True

    return out