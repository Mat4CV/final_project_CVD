import numpy as np
import plotly.graph_objects as go


def plot_fourier_spectrum_3d(
    spectrum: np.ndarray,
    mode: str = "scatter",
    threshold_percentile: float = 80.0,
    colorscale: str = "Viridis",
    title: str = "3D Fourier Spectrum",
) -> go.Figure:
    """
    Visualize a 3D Fourier spectrum using Plotly.

    Parameters
    ----------
    spectrum : np.ndarray
        Complex 3D array (output of np.fft.fftn or similar).
        Shape: (Nx, Ny, Nz)
    mode : str
        'scatter' — scatter plot where dot size & opacity encode log magnitude.
        'voxel'   — volume/isosurface rendering where opacity encodes log magnitude.
    threshold_percentile : float
        Only points/voxels above this percentile of log-magnitude are shown.
        Useful to declutter the plot (default: 80 → top 20% shown).
    colorscale : str
        Plotly colorscale name (e.g. 'Viridis', 'Plasma', 'Turbo', 'Hot').
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if spectrum.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {spectrum.shape}")

    Nx, Ny, Nt = spectrum.shape

    # ── Frequency axes (fftshift so DC is at centre) ──────────────────────────
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0)) * Nx   # cycles, step = 1
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0)) * Ny
    ft = np.fft.fftshift(np.fft.fftfreq(Nt, d=1.0)) * Nt

    # ── Shift spectrum so DC is at centre ─────────────────────────────────────
    shifted = np.fft.fftshift(spectrum)

    # ── Log magnitude ─────────────────────────────────────────────────────────
    magnitude = np.abs(shifted)
    log_mag = np.log1p(magnitude)           # log(1 + |F|) — safe for zeros

    # ── Threshold mask ────────────────────────────────────────────────────────
    thresh = np.percentile(log_mag, threshold_percentile)
    mask = log_mag >= thresh

    # ── Build coordinate grids ────────────────────────────────────────────────
    FX, FY, FT = np.meshgrid(fx, fy, ft, indexing="ij")

    # ── Normalised log-magnitude [0, 1] for colour / opacity / size ───────────
    lm_flat = log_mag[mask]
    lm_min, lm_max = lm_flat.min(), lm_flat.max()
    lm_norm = (lm_flat - lm_min) / (lm_max - lm_min + 1e-12)

    if mode == "scatter":
        fig = _scatter_plot(
            FX[mask], FY[mask], FT[mask],
            lm_norm, lm_flat,
            colorscale,
        )
    elif mode == "voxel":
        fig = _volume_plot(
            FX, FY, FT, log_mag,
            thresh, colorscale,
        )
    else:
        raise ValueError(f"mode must be 'scatter' or 'voxel', got '{mode}'")

    # ── Axis labels / limits ──────────────────────────────────────────────────
    half = (
        np.array([Nx, Ny, Nt]) // 2
    )
    ax_range = lambda h: [-(h), h]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis=dict(
                title="Frequency X (step=1)",
                range=ax_range(half[0]),
            ),
            yaxis=dict(
                title="Frequency Y (step=1)",
                range=ax_range(half[1]),
            ),
            zaxis=dict(
                title="Frequency T (step=1)",
                range=ax_range(half[2]),
            ),
            bgcolor="rgb(10,10,20)",
            xaxis_backgroundcolor="rgb(15,15,30)",
            yaxis_backgroundcolor="rgb(15,15,30)",
            zaxis_backgroundcolor="rgb(15,15,30)",
        ),
        paper_bgcolor="rgb(10,10,20)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ── Scatter helper ────────────────────────────────────────────────────────────

def _scatter_plot(x, y, z, lm_norm, lm_raw, colorscale):
    """Scatter3d: colour, opacity, and marker size all scale with log magnitude."""
    # Marker size: map [0,1] → [2, 20] px
    sizes = 1 + (lm_norm ** 2) * 19

    # Per-point opacity: map [0,1] → [0.15, 1.0]
    opacities = 0.15 + lm_norm * 0.85

    # Plotly Scatter3d needs a single opacity scalar; we encode it via the
    # alpha channel by building an rgba colour array.
    import plotly.colors as pc

    # Sample the colorscale at each normalised value
    rgb_list = pc.sample_colorscale(colorscale, lm_norm.tolist())

    def rgba(rgb_str, alpha):
        """'rgb(r,g,b)' → 'rgba(r,g,b,a)'"""
        inner = rgb_str[4:-1]          # strip 'rgb(' and ')'
        return f"rgba({inner},{alpha:.3f})"

    colors = [rgba(rgb, a) for rgb, a in zip(rgb_list, opacities)]

    trace = go.Scatter3d(
        x=x.ravel(), y=y.ravel(), z=z.ravel(),
        mode="markers",
        marker=dict(
            size=sizes,
            color=lm_raw,               # drives the colorbar
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="log(1+|F|)"),
            opacity=1.0,                # override below via line trick
            line=dict(width=0),
        ),
        # Override colours with per-point rgba for true per-point alpha
        # Plotly supports this via marker.color as a list of rgba strings
        text=[f"log|F|={v:.3f}" for v in lm_raw],
        hovertemplate="x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<br>%{text}<extra></extra>",
    )
    # Apply per-point rgba colours (overrides colorscale but keeps colorbar)
    trace.marker.color = colors
    trace.marker.colorscale = None
    trace.marker.showscale = False

    fig = go.Figure(data=[trace])
    return fig


# ── Volume helper ─────────────────────────────────────────────────────────────

def _volume_plot(FX, FY, FZ, log_mag, thresh, colorscale):
    """
    Volume rendering: opacity proportional to log magnitude.
    Uses go.Volume for smooth voxel-like rendering.
    """
    lm_min = log_mag.min()
    lm_max = log_mag.max()

    # go.Volume expects flattened 1-D arrays
    trace = go.Volume(
        x=FX.ravel(),
        y=FY.ravel(),
        z=FZ.ravel(),
        value=log_mag.ravel(),
        isomin=thresh,
        isomax=lm_max,
        opacity=0.15,           # base opacity (blended per voxel by Plotly)
        surface_count=20,       # number of isosurface layers → more = smoother
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="log(1+|F|)"),
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    fig = go.Figure(data=[trace])
    return fig


# ── Velocity plane ────────────────────────────────────────────────────────────

def plot_velocity_plane(
    spectrum: np.ndarray,
    vx: float,
    vy: float,
    colorscale: str = "Hot",
    title: str | None = None,
    plane_opacity: float = 0.18,
    plane_color: str = "rgba(80,160,255,0.18)",
) -> go.Figure:
    """
    Overlay the constraint plane  vx·fx + vy·fy + ft = 0  on the 3-D Fourier
    spectrum scatter plot.

    The spectrum axes are  (fx, fy, ft)  where the third axis is *time*
    frequency.  Shape of ``spectrum``:  (Nx, Ny, Nt).

    The plane passes through the origin with normal  n = (vx, vy, 1).
    We render it as a ``go.Surface`` mesh spanning the visible frequency box,
    and add the spectrum scatter on top so you can see which coefficients lie
    on (or near) the plane.

    Parameters
    ----------
    spectrum : np.ndarray, complex, shape (Nx, Ny, Nt)
    vx, vy   : float  — spatial velocity components (normalised units)
    colorscale : Plotly colorscale for the spectrum scatter
    title    : figure title (auto-generated if None)
    plane_opacity : float in [0,1] — how transparent the plane surface is
    plane_color   : CSS rgba string for the plane mesh fill

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if spectrum.ndim != 3:
        raise ValueError(f"Expected shape (Nx, Ny, Nt), got {spectrum.shape}")

    Nx, Ny, Nt = spectrum.shape

    # ── Frequency axes, shifted so DC is at centre ────────────────────────────
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0)) * Nx
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0)) * Ny
    ft = np.fft.fftshift(np.fft.fftfreq(Nt, d=1.0)) * Nt

    shifted   = np.fft.fftshift(spectrum)
    magnitude = np.abs(shifted)
    log_mag   = np.log1p(magnitude)

    FX, FY, FT = np.meshgrid(fx, fy, ft, indexing="ij")

    # ── Scatter: show all points, opacity/size/colour by log-magnitude ─────────
    lm_all  = log_mag.ravel()
    lm_min, lm_max = lm_all.min(), lm_all.max()
    lm_norm = (lm_all - lm_min) / (lm_max - lm_min + 1e-12)

    sizes     = 1.5 + lm_norm * 10
    opacities = 0.05 + lm_norm * 0.75

    import plotly.colors as pc
    rgb_list = pc.sample_colorscale(colorscale, lm_norm.tolist())

    def rgba(rgb_str, alpha):
        inner = rgb_str[4:-1]
        return f"rgba({inner},{alpha:.3f})"

    colors = [rgba(r, a) for r, a in zip(rgb_list, opacities)]

    scatter = go.Scatter3d(
        x=FX.ravel(), y=FY.ravel(), z=FT.ravel(),
        mode="markers",
        name="Spectrum",
        marker=dict(size=sizes, color=colors, line=dict(width=0)),
        hovertemplate=(
            "fx=%{x:.1f}  fy=%{y:.1f}  ft=%{z:.1f}<br>"
            "log|F|=%{text}<extra></extra>"
        ),
        text=[f"{v:.3f}" for v in lm_all],
    )

    # ── Plane  ft = -(vx·fx + vy·fy) ─────────────────────────────────────────
    # Span the plane over the visible fx/fy box; clip ft to the ft axis range.
    hx, hy, ht = Nx // 2, Ny // 2, Nt // 2
    px = np.linspace(-hx, hx, 60)
    py = np.linspace(-hy, hy, 60)
    PX, PY = np.meshgrid(px, py)
    PT = -(vx * PX + vy * PY)
    PT = np.clip(PT, -ht, ht)          # keep inside the rendered volume

    plane = go.Surface(
        x=PX, y=PY, z=PT,
        opacity=plane_opacity,
        colorscale=[[0, plane_color], [1, plane_color]],
        showscale=False,
        name=f"Plane  {vx:+.2f}·fx + {vy:+.2f}·fy + ft = 0",
        hoverinfo="skip",
        contours=dict(
            x=dict(show=True, color="rgba(120,180,255,0.35)", width=1),
            y=dict(show=True, color="rgba(120,180,255,0.35)", width=1),
        ),
    )

    # ── Points on/near the plane ───────────────────────────────────────────────
    # Distance from each (fx, fy, ft) grid point to the plane, measured along ft
    residual = np.abs(FT + vx * FX + vy * FY)   # |ft + vx·fx + vy·fy|
    tol      = max(1.0, 0.5 * min(Nx, Ny, Nt) / 16)   # ≈ half a step
    on_plane = residual <= tol

    lm_plane  = log_mag[on_plane]
    lm_p_norm = (lm_plane - lm_min) / (lm_max - lm_min + 1e-12)
    sizes_p   = 3 + lm_p_norm * 14
    opac_p    = 0.5 + lm_p_norm * 0.5
    rgb_p     = pc.sample_colorscale("Plasma", lm_p_norm.tolist())
    colors_p  = [rgba(r, a) for r, a in zip(rgb_p, opac_p)]

    on_scatter = go.Scatter3d(
        x=FX[on_plane], y=FY[on_plane], z=FT[on_plane],
        mode="markers",
        name="On plane",
        marker=dict(size=sizes_p, color=colors_p, line=dict(width=0)),
        hovertemplate=(
            "fx=%{x:.1f}  fy=%{y:.1f}  ft=%{z:.1f}<br>"
            "log|F|=%{text}<extra></extra>"
        ),
        text=[f"{v:.3f}" for v in lm_plane],
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    auto_title = (
        title if title else
        f"Velocity plane  {vx:+.2f}·f_x + {vy:+.2f}·f_y + f_t = 0"
    )
    # fig = go.Figure(data=[scatter, plane, on_scatter])
    fig = go.Figure(data=[plane])
    fig.update_layout(
        title=dict(text=auto_title, font=dict(size=17)),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(20,20,40,0.7)",
            font=dict(color="white", size=11),
        ),
        scene=dict(
            xaxis=dict(title="f_x  (step=1)", range=[-hx, hx]),
            yaxis=dict(title="f_y  (step=1)", range=[-hy, hy]),
            zaxis=dict(title="f_t  (step=1)", range=[-ht, ht]),
            bgcolor="rgb(10,10,20)",
            xaxis_backgroundcolor="rgb(15,15,30)",
            yaxis_backgroundcolor="rgb(15,15,30)",
            zaxis_backgroundcolor="rgb(15,15,30)",
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
        ),
        paper_bgcolor="rgb(10,10,20)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
