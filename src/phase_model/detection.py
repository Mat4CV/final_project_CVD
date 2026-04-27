import itertools

import numpy as np


def frequency_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the 2D angular-frequency grid used by the Fourier phase model.

    Convention:
        x = image columns
        y = image rows
        vx = horizontal velocity in pixels/frame
        vy = vertical velocity in pixels/frame

    Returns:
        wx, wy:
            Arrays of shape H x W.
    """
    wx_1d = 2.0 * np.pi * np.fft.fftfreq(w)
    wy_1d = 2.0 * np.pi * np.fft.fftfreq(h)

    wx, wy = np.meshgrid(wx_1d, wy_1d)
    return wx, wy


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Wrapped angular distance in radians.

    This returns values in [0, pi].
    """
    return np.abs(np.angle(np.exp(1j * (a - b))))


def solve_two_phasors_closed_form(
    F_seq: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form Fourier Vision Chapter 3 solver for two additive components.

    At each spatial frequency, the model is:

        F_t = A_1 z_1^t + A_2 z_2^t

    where:
        z_i = exp(-j (wx vx_i + wy vy_i) dt)

    Args:
        F_seq:
            Complex Fourier sequence with shape T x H x W.
            Only the first four frames are used.
        eps:
            Numerical threshold.

    Returns:
        z:
            Complex phase steps with shape 2 x H x W.
        A:
            Complex component Fourier coefficients at t=0 with shape 2 x H x W.
        valid:
            Boolean valid-frequency mask with shape H x W.
    """
    if F_seq.ndim != 3:
        raise ValueError("F_seq must have shape T x H x W.")

    if F_seq.shape[0] < 4:
        raise ValueError("The m=2 closed-form solver requires at least four frames.")

    F0, F1, F2, F3 = F_seq[:4]

    a = F1**2 - F0 * F2
    b = F0 * F3 - F1 * F2
    c = F2**2 - F1 * F3

    discriminant = b**2 - 4.0 * a * c
    sqrt_discriminant = np.sqrt(discriminant)

    z1 = np.zeros_like(F0, dtype=np.complex128)
    z2 = np.zeros_like(F0, dtype=np.complex128)
    A1 = np.zeros_like(F0, dtype=np.complex128)
    A2 = np.zeros_like(F0, dtype=np.complex128)

    valid = np.abs(a) > eps

    z1[valid] = (-b[valid] + sqrt_discriminant[valid]) / (2.0 * a[valid])
    z2[valid] = (-b[valid] - sqrt_discriminant[valid]) / (2.0 * a[valid])

    denom = z2 - z1
    valid &= np.abs(denom) > eps

    A1[valid] = (F0[valid] * z2[valid] - F1[valid]) / denom[valid]
    A2[valid] = (F0[valid] * z1[valid] - F1[valid]) / (-denom[valid])

    valid &= np.isfinite(z1)
    valid &= np.isfinite(z2)
    valid &= np.isfinite(A1)
    valid &= np.isfinite(A2)

    z = np.stack([z1, z2], axis=0)
    A = np.stack([A1, A2], axis=0)

    z_abs = np.abs(z)
    valid &= np.all(z_abs > eps, axis=0)

    # In the ideal model the roots live on the unit circle.
    # Normalizing makes the subsequent phase/Hough step more stable.
    z[:, valid] = z[:, valid] / np.abs(z[:, valid])

    return z, A, valid


def solve_phasors_prony(
    F_seq: np.ndarray,
    num_components: int,
    eps: float = 1e-8,
    normalize_roots: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic Prony / annihilating-filter solver for m additive components.

    At each spatial frequency, the model is:

        F_t = sum_i A_i z_i^t

    with i = 1, ..., m.

    This extends the m=2 closed-form solver to m=3,4,...

    Args:
        F_seq:
            Complex Fourier sequence with shape T x H x W.
        num_components:
            Number of additive translating components.
        eps:
            Numerical threshold.
        normalize_roots:
            If True, project estimated roots z_i onto the unit circle.

    Returns:
        z:
            Complex phase steps with shape m x H x W.
        A:
            Complex component Fourier coefficients at t=0 with shape m x H x W.
        valid:
            Boolean valid-frequency mask with shape H x W.

    Notes:
        Requires T >= 2m.
    """
    if F_seq.ndim != 3:
        raise ValueError("F_seq must have shape T x H x W.")

    T, H, W = F_seq.shape
    m = int(num_components)

    if m < 1:
        raise ValueError("num_components must be at least 1.")

    if T < 2 * m:
        raise ValueError(f"Need at least 2m frames. Got T={T}, m={m}.")

    z_out = np.zeros((m, H, W), dtype=np.complex128)
    A_out = np.zeros((m, H, W), dtype=np.complex128)
    valid = np.zeros((H, W), dtype=bool)

    # Recurrence:
    #
    #   F_{t+m} + c_{m-1} F_{t+m-1} + ... + c_1 F_{t+1} + c_0 F_t = 0
    #
    # Number of available recurrence equations.
    num_eqs = T - m

    for y in range(H):
        for x in range(W):
            s = F_seq[:, y, x].astype(np.complex128, copy=False)

            if not np.all(np.isfinite(s)):
                continue

            if np.linalg.norm(s) < eps:
                continue

            M = np.empty((num_eqs, m), dtype=np.complex128)
            rhs = np.empty(num_eqs, dtype=np.complex128)

            for t in range(num_eqs):
                M[t, :] = s[t : t + m]
                rhs[t] = -s[t + m]

            try:
                coeffs, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=None)
            except np.linalg.LinAlgError:
                continue

            if rank < m:
                continue

            # Polynomial:
            #
            #   z^m + c_{m-1} z^{m-1} + ... + c_1 z + c_0
            #
            # np.roots expects coefficients from highest to lowest degree.
            poly_coeffs = np.concatenate(([1.0 + 0.0j], coeffs[::-1]))
            roots = np.roots(poly_coeffs)

            if roots.shape[0] != m:
                continue

            if not np.all(np.isfinite(roots)):
                continue

            root_abs = np.abs(roots)

            if np.any(root_abs < eps):
                continue

            if normalize_roots:
                roots = roots / root_abs

            V = np.empty((T, m), dtype=np.complex128)

            for t in range(T):
                V[t, :] = roots**t

            try:
                amps, _, rank, _ = np.linalg.lstsq(V, s, rcond=None)
            except np.linalg.LinAlgError:
                continue

            if rank < m:
                continue

            if not np.all(np.isfinite(amps)):
                continue

            z_out[:, y, x] = roots
            A_out[:, y, x] = amps
            valid[y, x] = True

    return z_out, A_out, valid


def phase_hough_votes(
    z_fields: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    valid: np.ndarray,
    vx_values: np.ndarray,
    vy_values: np.ndarray,
    dt: float = 1.0,
    weights: np.ndarray | None = None,
    sigma: float = 0.15,
) -> np.ndarray:
    """
    Generic Hough voting from phase-step fields.

    For a candidate velocity (vx, vy), the predicted phase step is:

        phase = -(wx vx + wy vy) dt

    Args:
        z_fields:
            Complex phase steps with shape m x H x W.
        wx, wy:
            Angular frequency grids with shape H x W.
        valid:
            Boolean valid-frequency mask with shape H x W.
        vx_values:
            Candidate horizontal velocities.
        vy_values:
            Candidate vertical velocities.
        dt:
            Time step between frames.
        weights:
            Optional weights with shape m x H x W.
            Usually abs(A), where A are the recovered Fourier amplitudes.
        sigma:
            Angular tolerance for soft voting.

    Returns:
        Hough accumulator with shape len(vy_values) x len(vx_values).
    """
    if z_fields.ndim != 3:
        raise ValueError("z_fields must have shape m x H x W.")

    m, H, W = z_fields.shape

    if wx.shape != (H, W) or wy.shape != (H, W):
        raise ValueError("wx and wy must have shape H x W.")

    if valid.shape != (H, W):
        raise ValueError("valid must have shape H x W.")

    if weights is None:
        weights = np.ones_like(z_fields, dtype=np.float64)
    elif weights.shape != z_fields.shape:
        raise ValueError("weights must have the same shape as z_fields.")

    accumulator = np.zeros((len(vy_values), len(vx_values)), dtype=np.float64)

    wx_flat = wx[valid]
    wy_flat = wy[valid]

    if wx_flat.size == 0:
        return accumulator

    for branch in range(m):
        phase_obs = np.angle(z_fields[branch])[valid]
        weight_flat = weights[branch][valid]

        for iy, vy in enumerate(vy_values):
            for ix, vx in enumerate(vx_values):
                phase_pred = -(wx_flat * vx + wy_flat * vy) * dt
                err = angular_distance(phase_obs, phase_pred)
                vote = np.exp(-(err**2) / (2.0 * sigma**2))
                accumulator[iy, ix] += np.sum(weight_flat * vote)

    return accumulator


def top_k_peaks(
    accumulator: np.ndarray,
    vx_values: np.ndarray,
    vy_values: np.ndarray,
    k: int,
    min_separation: int = 5,
) -> list[tuple[float, float]]:
    """
    Select the top-k local maxima from a velocity accumulator.

    Args:
        accumulator:
            Hough accumulator with shape len(vy_values) x len(vx_values).
        vx_values:
            Candidate horizontal velocities.
        vy_values:
            Candidate vertical velocities.
        k:
            Number of peaks to return.
        min_separation:
            Suppression radius in grid cells after selecting a peak.

    Returns:
        List of velocities [(vx, vy), ...].
    """
    if accumulator.shape != (len(vy_values), len(vx_values)):
        raise ValueError("accumulator shape must be len(vy_values) x len(vx_values).")

    acc = accumulator.astype(np.float64, copy=True)
    velocities: list[tuple[float, float]] = []

    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(acc), acc.shape)

        if not np.isfinite(acc[iy, ix]):
            break

        velocities.append((float(vx_values[ix]), float(vy_values[iy])))

        y0 = max(0, iy - min_separation)
        y1 = min(acc.shape[0], iy + min_separation + 1)
        x0 = max(0, ix - min_separation)
        x1 = min(acc.shape[1], ix + min_separation + 1)

        acc[y0:y1, x0:x1] = -np.inf

    return velocities


def expected_phase_step(
    wx: np.ndarray,
    wy: np.ndarray,
    velocity: tuple[float, float],
    dt: float = 1.0,
) -> np.ndarray:
    """
    Expected complex phase step for a given velocity.
    """
    vx, vy = velocity
    phase = -(wx * vx + wy * vy) * dt
    return np.exp(1j * phase)


def sort_components_by_velocity(
    A: np.ndarray,
    z: np.ndarray,
    velocities: list[tuple[float, float]],
    wx: np.ndarray,
    wy: np.ndarray,
    valid: np.ndarray,
    dt: float = 1.0,
) -> list[np.ndarray]:
    """
    Sort unordered phasor branches into object-specific Fourier spectra.

    Args:
        A:
            Component amplitudes with shape m x H x W.
        z:
            Phase steps with shape m x H x W.
        velocities:
            Estimated velocities. Must have length m.
        wx, wy:
            Frequency grids with shape H x W.
        valid:
            Boolean mask with shape H x W.
        dt:
            Time step.

    Returns:
        spectra:
            List of m arrays, each with shape H x W.

    Notes:
        At each frequency, Prony gives unordered branches.
        This function assigns branches to objects by minimizing phase error
        against the expected phase induced by each detected velocity.
    """
    if A.shape != z.shape:
        raise ValueError("A and z must have the same shape.")

    if z.ndim != 3:
        raise ValueError("A and z must have shape m x H x W.")

    m, H, W = z.shape

    if len(velocities) != m:
        raise ValueError("Need exactly one velocity per component to sort spectra.")

    if wx.shape != (H, W) or wy.shape != (H, W):
        raise ValueError("wx and wy must have shape H x W.")

    if valid.shape != (H, W):
        raise ValueError("valid must have shape H x W.")

    spectra = [np.zeros((H, W), dtype=np.complex128) for _ in range(m)]

    expected_phases = np.empty((m, H, W), dtype=np.float64)

    for obj, velocity in enumerate(velocities):
        vx, vy = velocity
        expected_phases[obj] = -(wx * vx + wy * vy) * dt

    observed_phases = np.angle(z)

    for y in range(H):
        for x in range(W):
            if not valid[y, x]:
                continue

            cost = np.empty((m, m), dtype=np.float64)

            for branch in range(m):
                for obj in range(m):
                    cost[branch, obj] = angular_distance(
                        observed_phases[branch, y, x],
                        expected_phases[obj, y, x],
                    )

            best_perm = None
            best_cost = np.inf

            for perm in itertools.permutations(range(m)):
                total = 0.0

                for branch, obj in enumerate(perm):
                    total += cost[branch, obj]

                if total < best_cost:
                    best_cost = total
                    best_perm = perm

            for branch, obj in enumerate(best_perm):
                spectra[obj][y, x] = A[branch, y, x]

    return spectra


def reconstruct_from_spectra(spectra: list[np.ndarray]) -> list[np.ndarray]:
    """
    Reconstruct spatial-domain component images from Fourier spectra.
    """
    return [np.real(np.fft.ifft2(F)) for F in spectra]