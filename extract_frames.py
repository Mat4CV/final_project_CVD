#!/usr/bin/env python3
"""
Extract frames from all MP4 files in a given folder.
Each MP4 gets its own subfolder containing all extracted frames.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_frames(mp4_path: Path, output_dir: Path) -> int:
    """Extract all frames from an MP4 file into output_dir using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.png"

    cmd = [
        "ffmpeg",
        "-i", str(mp4_path),
        "-vsync", "0",          # Keep every frame exactly once
        "-q:v", "2",            # High quality
        str(output_pattern),
        "-y",                   # Overwrite without asking
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed:\n{result.stderr.strip()}", file=sys.stderr)
        return 0

    return len(list(output_dir.glob("frame_*.png")))


def process_folder(folder: Path) -> None:
    mp4_files = sorted(folder.glob("*.mp4"))

    if not mp4_files:
        print(f"No MP4 files found in: {folder}")
        return

    print(f"Found {len(mp4_files)} MP4 file(s) in '{folder}'\n")

    for mp4 in mp4_files:
        output_dir = folder / mp4.stem          # Folder named after the MP4 (no extension)
        print(f"Processing: {mp4.name}")
        print(f"  Output folder: {output_dir}")

        count = extract_frames(mp4, output_dir)
        if count:
            print(f"  Extracted {count} frame(s)\n")
        else:
            print(f"  No frames extracted (check ffmpeg output above)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from all MP4 files in a folder."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the folder containing MP4 files",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()

    if not folder.exists():
        print(f"Error: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    # Check ffmpeg is available
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        print("Error: ffmpeg is not installed or not on PATH.", file=sys.stderr)
        print("Install it with:  brew install ffmpeg  /  apt install ffmpeg", file=sys.stderr)
        sys.exit(1)

    process_folder(folder)


if __name__ == "__main__":
    main()
