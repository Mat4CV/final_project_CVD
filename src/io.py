"""
Utilities for Milestone 1: video loading and quick visualization.

Main features
-------------
- Load a video file into a NumPy array.
- Optionally convert to grayscale.
- Optionally center-crop and resize.
- Save sample frames for debugging.
- Save a GIF or MP4 preview for quick inspection.

Conventions
-----------
- Color videos are returned as (T, H, W, C) with C=3 in RGB order.
- Grayscale videos are returned as (T, H, W).
- Pixel values are uint8 in [0, 255].

Recommended use
---------------
This file is meant to be the first stable I/O utility for the project.
Everything later (synthetic tests, FFT pipeline, compensation, etc.)
should reuse these helpers instead of re-implementing loading logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "src/io.py requires OpenCV. Install it with:\n"
        "  uv add opencv-python\n"
        "or\n"
        "  pip install opencv-python"
    ) from e


ArrayLikeVideo = np.ndarray


@dataclass
class VideoInfo:
    """Basic metadata for a loaded video."""
    path: str
    num_frames: int
    height: int
    width: int
    channels: int
    fps: float


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _center_crop_frame(frame: np.ndarray, crop_hw: Tuple[int, int]) -> np.ndarray:
    """
    Center crop a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Shape (H, W) or (H, W, C).
    crop_hw : tuple[int, int]
        Desired (crop_h, crop_w).

    Returns
    -------
    np.ndarray
        Cropped frame.
    """
    crop_h, crop_w = crop_hw
    h, w = frame.shape[:2]

    if crop_h > h or crop_w > w:
        raise ValueError(
            f"Crop size {(crop_h, crop_w)} is larger than frame size {(h, w)}."
        )

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return frame[top:top + crop_h, left:left + crop_w]


def _resize_frame(
    frame: np.ndarray,
    resize_hw: Tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Shape (H, W) or (H, W, C).
    resize_hw : tuple[int, int]
        Desired (new_h, new_w).

    Returns
    -------
    np.ndarray
        Resized frame.
    """
    new_h, new_w = resize_hw
    # cv2.resize expects (width, height)
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)


def _to_grayscale(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 frame to grayscale.

    Parameters
    ----------
    frame_rgb : np.ndarray
        Shape (H, W, 3) in RGB order.

    Returns
    -------
    np.ndarray
        Shape (H, W), uint8 grayscale.
    """
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)


def load_video(
    video_path: str | Path,
    grayscale: bool = False,
    crop_hw: Optional[Tuple[int, int]] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> tuple[ArrayLikeVideo, VideoInfo]:
    """
    Load a video from disk into a NumPy array.

    Parameters
    ----------
    video_path : str or Path
        Path to input video.
    grayscale : bool, default=False
        If True, return shape (T, H, W). Otherwise return (T, H, W, 3).
    crop_hw : tuple[int, int] or None
        Optional center crop size (crop_h, crop_w).
    resize_hw : tuple[int, int] or None
        Optional resize target (new_h, new_w).
    max_frames : int or None
        Maximum number of frames to load.
    start_frame : int, default=0
        Index of first frame to read.

    Returns
    -------
    video : np.ndarray
        Video array. Shape is (T, H, W) for grayscale or (T, H, W, 3) for RGB.
    info : VideoInfo
        Basic video metadata.

    Notes
    -----
    Frames are returned as uint8 in RGB order for color videos.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if start_frame >= total_frames and total_frames > 0:
        raise ValueError(
            f"start_frame={start_frame} is beyond the video length ({total_frames} frames)."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    frames_read = 0

    while True:
        if max_frames is not None and frames_read >= max_frames:
            break

        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if crop_hw is not None:
            frame_rgb = _center_crop_frame(frame_rgb, crop_hw)

        if resize_hw is not None:
            frame_rgb = _resize_frame(frame_rgb, resize_hw)

        if grayscale:
            frame = _to_grayscale(frame_rgb)
        else:
            frame = frame_rgb

        frames.append(frame)
        frames_read += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames were loaded from {video_path}")

    video = np.stack(frames, axis=0)
    h, w = video.shape[1:3]
    c = 1 if video.ndim == 3 else video.shape[3]

    info = VideoInfo(
        path=video_path,
        num_frames=video.shape[0],
        height=h,
        width=w,
        channels=c,
        fps=fps,
    )
    return video, info


def save_sample_frames(
    video: np.ndarray,
    output_dir: str | Path,
    frame_indices: Optional[Sequence[int]] = None,
    prefix: str = "frame",
) -> None:
    """
    Save selected frames as PNG images.

    Parameters
    ----------
    video : np.ndarray
        Shape (T, H, W) or (T, H, W, C).
    output_dir : str or Path
        Folder where images are saved.
    frame_indices : sequence[int] or None
        Indices to save. If None, save [0, T//2, T-1].
    prefix : str
        Output filename prefix.
    """
    output_dir = _ensure_dir(output_dir)
    T = video.shape[0]

    if frame_indices is None:
        frame_indices = sorted(set([0, T // 2, T - 1]))

    for idx in frame_indices:
        if idx < 0 or idx >= T:
            raise IndexError(f"Frame index {idx} is out of range for T={T}")

        frame = video[idx]
        out_path = output_dir / f"{prefix}_{idx:04d}.png"

        if frame.ndim == 2:
            cv2.imwrite(str(out_path), frame)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), frame_bgr)


def save_gif(
    video: np.ndarray,
    output_path: str | Path,
    fps: float = 12.0,
) -> None:
    """
    Save a video array as a GIF.

    Parameters
    ----------
    video : np.ndarray
        Shape (T, H, W) or (T, H, W, C), uint8.
    output_path : str or Path
        Destination .gif path.
    fps : float
        Playback frame rate for the GIF.
    """
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = 1.0 / fps
    frames = []

    for frame in video:
        if frame.ndim == 2:
            frames.append(frame)
        else:
            frames.append(frame)

    imageio.mimsave(output_path, frames, duration=duration)


def save_mp4(
    video: np.ndarray,
    output_path: str | Path,
    fps: float = 24.0,
) -> None:
    """
    Save a video array as an MP4.

    Parameters
    ----------
    video : np.ndarray
        Shape (T, H, W) or (T, H, W, C), uint8.
    output_path : str or Path
        Destination .mp4 path.
    fps : float
        Output frame rate.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = video.shape[0]
    h, w = video.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=True)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    for i in range(T):
        frame = video[i]
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def inspect_video(
    video: np.ndarray,
    info: Optional[VideoInfo] = None,
    name: str = "video",
) -> None:
    """
    Print a short summary of a loaded video.

    Parameters
    ----------
    video : np.ndarray
        Loaded video.
    info : VideoInfo or None
        Optional metadata.
    name : str
        Display name.
    """
    print(f"[{name}]")
    print(f"  shape      : {video.shape}")
    print(f"  dtype      : {video.dtype}")
    print(f"  min / max  : {video.min()} / {video.max()}")

    if info is not None:
        print(f"  path       : {info.path}")
        print(f"  fps        : {info.fps:.3f}")
        print(f"  frames     : {info.num_frames}")
        print(f"  height     : {info.height}")
        print(f"  width      : {info.width}")
        print(f"  channels   : {info.channels}")


def load_and_preview(
    video_path: str | Path,
    output_dir: str | Path,
    grayscale: bool = False,
    crop_hw: Optional[Tuple[int, int]] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    preview_fps: float = 12.0,
) -> tuple[np.ndarray, VideoInfo]:
    """
    Convenience function for Milestone 1.

    It:
    1. loads the video
    2. prints summary info
    3. saves sample frames
    4. saves a GIF preview
    5. saves an MP4 preview

    Returns
    -------
    video, info
    """
    output_dir = _ensure_dir(output_dir)

    video, info = load_video(
        video_path=video_path,
        grayscale=grayscale,
        crop_hw=crop_hw,
        resize_hw=resize_hw,
        max_frames=max_frames,
        start_frame=start_frame,
    )

    inspect_video(video, info)

    save_sample_frames(video, output_dir / "frames")
    save_gif(video, output_dir / "preview.gif", fps=preview_fps)
    save_mp4(video, output_dir / "preview.mp4", fps=preview_fps)

    return video, info


if __name__ == "__main__":
    # Minimal example:
    #
    #   python src/io.py path/to/video.mp4
    #
    import argparse

    parser = argparse.ArgumentParser(description="Load and preview a video.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="results/io_demo")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--crop_h", type=int, default=None)
    parser.add_argument("--crop_w", type=int, default=None)
    parser.add_argument("--resize_h", type=int, default=None)
    parser.add_argument("--resize_w", type=int, default=None)
    parser.add_argument("--preview_fps", type=float, default=12.0)

    args = parser.parse_args()

    crop_hw = None
    if args.crop_h is not None and args.crop_w is not None:
        crop_hw = (args.crop_h, args.crop_w)

    resize_hw = None
    if args.resize_h is not None and args.resize_w is not None:
        resize_hw = (args.resize_h, args.resize_w)

    load_and_preview(
        video_path=args.video_path,
        output_dir=args.output_dir,
        grayscale=args.grayscale,
        crop_hw=crop_hw,
        resize_hw=resize_hw,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        preview_fps=args.preview_fps,
    )