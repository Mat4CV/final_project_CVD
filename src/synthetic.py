"""
Synthetic video generation for Fourier-domain motion estimation.

Coordinate convention:
    video[t, y, x]
    center = (cx, cy)
    velocity = (vx, vy) in pixels/frame

So:
    vx > 0 moves right
    vy > 0 moves down
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MovingObject:
    kind: str  # "gaussian", "disk", or "square"
    center: tuple[float, float]  # (cx, cy)
    velocity: tuple[float, float]  # (vx, vy), pixels/frame
    size: float
    amplitude: float = 1.0


def _render_gaussian(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2)
    )


def _render_disk(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    amplitude: float,
) -> np.ndarray:
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius**2
    return amplitude * mask.astype(np.float32)


def _render_square(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    half_size: float,
    amplitude: float,
) -> np.ndarray:
    mask = (np.abs(X - cx) <= half_size) & (np.abs(Y - cy) <= half_size)
    return amplitude * mask.astype(np.float32)


def _render_object(
    obj: MovingObject,
    X: np.ndarray,
    Y: np.ndarray,
    t: int,
) -> np.ndarray:
    cx0, cy0 = obj.center
    vx, vy = obj.velocity

    cx = cx0 + vx * t
    cy = cy0 + vy * t

    if obj.kind == "gaussian":
        return _render_gaussian(X, Y, cx, cy, obj.size, obj.amplitude)

    if obj.kind == "disk":
        return _render_disk(X, Y, cx, cy, obj.size, obj.amplitude)

    if obj.kind == "square":
        return _render_square(X, Y, cx, cy, obj.size, obj.amplitude)

    raise ValueError(
        f"Unknown object kind '{obj.kind}'. "
        "Valid choices are: 'gaussian', 'disk', 'square'."
    )


def generate_synthetic_sequence(
    T: int = 64,
    H: int = 128,
    W: int = 128,
    objects: list[MovingObject] | None = None,
    background: float = 0.0,
    noise_std: float = 0.0,
    normalize: bool = False,
    clip: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate a synthetic video with one or more translating objects.

    Args:
        T: Number of frames.
        H: Image height.
        W: Image width.
        objects: List of MovingObject instances.
        background: Constant background intensity.
        noise_std: Standard deviation of additive Gaussian noise.
        normalize: If True, linearly normalize video to [0, 1].
        clip: If True, clip video to [0, 1].
        seed: Random seed.

    Returns:
        video:
            Array of shape (T, H, W), dtype float32.
        masks:
            Array of shape (K, T, H, W), one binary-ish mask per object.
        metadata:
            Dictionary containing ground-truth parameters.
    """
    rng = np.random.default_rng(seed)

    if objects is None:
        objects = [
            MovingObject(
                kind="gaussian",
                center=(W / 3.0, H / 2.0),
                velocity=(1.0, 0.0),
                size=6.0,
                amplitude=1.0,
            )
        ]

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    video = np.full((T, H, W), background, dtype=np.float32)
    masks = np.zeros((len(objects), T, H, W), dtype=np.float32)

    for k, obj in enumerate(objects):
        for t in range(T):
            frame_obj = _render_object(obj, xx, yy, t).astype(np.float32)
            video[t] += frame_obj

            # Works for all object types, including Gaussian.
            threshold = 0.05 * obj.amplitude
            masks[k, t] = (frame_obj > threshold).astype(np.float32)

    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=video.shape).astype(np.float32)
        video += noise

    if normalize:
        vmin = float(video.min())
        vmax = float(video.max())
        eps = 1e-8
        video = (video - vmin) / (vmax - vmin + eps)

    if clip:
        video = np.clip(video, 0.0, 1.0)

    metadata = {
        "T": T,
        "H": H,
        "W": W,
        "background": background,
        "noise_std": noise_std,
        "normalize": normalize,
        "clip": clip,
        "seed": seed,
        "objects": [asdict(obj) for obj in objects],
        "velocities": [obj.velocity for obj in objects],
        "centers": [obj.center for obj in objects],
        "kinds": [obj.kind for obj in objects],
    }

    return video.astype(np.float32), masks.astype(np.float32), metadata


def save_video_summary(
    video: np.ndarray,
    out_path: str | Path = "results/synthetic_summary.png",
    title: str | None = None,
) -> None:
    """
    Save a row of representative frames.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"

    T = video.shape[0]
    indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))

    if len(indices) == 1:
        axes = [axes]

    for ax, t in zip(axes, indices):
        ax.imshow(video[t], cmap="gray", vmin=float(video.min()), vmax=float(video.max()))
        ax.set_title(f"t = {t}")
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sequence(
    video: np.ndarray,
    masks: np.ndarray,
    metadata: dict,
    out_dir: str | Path,
    name: str,
) -> None:
    """
    Save video, masks, and metadata.

    Produces:
        {out_dir}/{name}.npy
        {out_dir}/{name}_masks.npy
        {out_dir}/{name}_metadata.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{name}.npy", video)
    np.save(out_dir / f"{name}_masks.npy", masks)

    with open(out_dir / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def make_single_gaussian() -> tuple[np.ndarray, np.ndarray, dict]:
    objects = [
        MovingObject(
            kind="gaussian",
            center=(32.0, 64.0),
            velocity=(1.0, 0.0),
            size=6.0,
            amplitude=1.0,
        )
    ]

    return generate_synthetic_sequence(
        T=64,
        H=128,
        W=128,
        objects=objects,
        background=0.0,
        noise_std=0.0,
        seed=0,
    )


def make_single_disk() -> tuple[np.ndarray, np.ndarray, dict]:
    objects = [
        MovingObject(
            kind="disk",
            center=(32.0, 64.0),
            velocity=(1.0, 0.5),
            size=8.0,
            amplitude=1.0,
        )
    ]

    return generate_synthetic_sequence(
        T=64,
        H=128,
        W=128,
        objects=objects,
        background=0.0,
        noise_std=0.0,
        seed=1,
    )


def make_two_objects() -> tuple[np.ndarray, np.ndarray, dict]:
    objects = [
        MovingObject(
            kind="gaussian",
            center=(32.0, 48.0),
            velocity=(1.0, 0.0),
            size=6.0,
            amplitude=1.0,
        ),
        MovingObject(
            kind="disk",
            center=(96.0, 80.0),
            velocity=(-0.5, -0.75),
            size=8.0,
            amplitude=0.8,
        ),
    ]

    return generate_synthetic_sequence(
        T=64,
        H=128,
        W=128,
        objects=objects,
        background=0.0,
        noise_std=0.0,
        seed=2,
    )


def main() -> None:
    out_dir = Path("results/synthetic")

    datasets = {
        "single_gaussian": make_single_gaussian(),
        "single_disk": make_single_disk(),
        "two_objects": make_two_objects(),
    }

    for name, (video, masks, metadata) in datasets.items():
        save_sequence(video, masks, metadata, out_dir=out_dir, name=name)

        save_video_summary(
            video,
            out_path=out_dir / f"{name}_summary.png",
            title=name.replace("_", " ").title(),
        )

        print(f"Saved {name}")
        print(f"  video shape: {video.shape}")
        print(f"  masks shape: {masks.shape}")
        print(f"  velocities: {metadata['velocities']}")
        print()


if __name__ == "__main__":
    main()