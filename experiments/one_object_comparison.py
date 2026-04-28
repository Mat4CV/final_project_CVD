"""
Run one-object Fourier motion benchmarks.

Compares:
    1. Phase / Prony detector from fourier_vision
    2. 3D Fourier plane detector from plane_scoring_detection

Example:

    uv run python experiments/one_object_comparison.py \
        --out results/one_object_benchmark.csv \
        --methods phase plane \
        --shapes gaussian disk square \
        --speeds 0.5 1.0 1.5 2.0 \
        --directions 0 45 90 135 180 225 270 315 \
        --noise-levels 0.0 0.05 0.10 0.20 \
        --frame-counts 16 32 64 \
        --seeds 0 1 2 \
        --height 128 \
        --width 128 \
        --velocity-min -2.5 \
        --velocity-max 2.5 \
        --velocity-bins 151 \
        --num-workers 4 \
        --use-gpu

Resume example:

    uv run python experiments/one_object_comparison.py \
        --out results/one_object_gaussian.csv \
        --shapes gaussian \
        --resume \
        --use-gpu \
        --num-workers 1

Notes:
    - If you only have one GPU, do not set num-workers too high.
    - For CPU-only runs, omit --use-gpu.
    - With --resume, the script counts existing rows in the CSV and skips that
      many jobs from the deterministic job list.
    - Use --overwrite only when you intentionally want to delete the old CSV.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------
# Import path setup
# ----------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_PATH:", SRC_PATH)
print("SRC exists:", SRC_PATH.exists())


# ----------------------------
# Experiment specification
# ----------------------------

@dataclass(frozen=True)
class ExperimentJob:
    method: str
    shape: str
    size: float
    speed: float
    direction_deg: float
    noise_std: float
    T: int
    H: int
    W: int
    seed: int

    velocity_min: float
    velocity_max: float
    velocity_bins: int

    # Phase method knobs
    phase_hough_sigma: float
    phase_peak_min_separation: int
    phase_min_frequency_radius: float

    # Plane detector knobs
    plane_sigma: float
    plane_alpha: float
    plane_dc_bins: int
    plane_keep_frac: float
    plane_min_detection_separation: int

    # Runtime knobs
    use_gpu: bool
    gpu_ids: tuple[int, ...]


# ----------------------------
# Geometry / metrics
# ----------------------------

def velocity_from_speed_direction(
    speed: float,
    direction_deg: float,
) -> tuple[float, float]:
    theta = math.radians(direction_deg)
    vx = speed * math.cos(theta)
    vy = speed * math.sin(theta)
    return float(vx), float(vy)


def centered_trajectory_start(
    H: int,
    W: int,
    vx: float,
    vy: float,
    T: int,
) -> tuple[float, float]:
    """
    Choose initial center so the whole trajectory is centered in the image.

    Coordinate convention:
        center = (cx, cy)
        velocity = (vx, vy)
    """
    cx_mid = W / 2.0
    cy_mid = H / 2.0

    cx0 = cx_mid - 0.5 * vx * (T - 1)
    cy0 = cy_mid - 0.5 * vy * (T - 1)

    return float(cx0), float(cy0)


def endpoint_error(
    v_hat: tuple[float, float],
    v_gt: tuple[float, float],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(v_hat, dtype=float) - np.asarray(v_gt, dtype=float)
        )
    )


def speed_error(
    v_hat: tuple[float, float],
    v_gt: tuple[float, float],
) -> float:
    return float(abs(np.linalg.norm(v_hat) - np.linalg.norm(v_gt)))


def angular_error_deg(
    v_hat: tuple[float, float],
    v_gt: tuple[float, float],
) -> float:
    v_hat_arr = np.asarray(v_hat, dtype=float)
    v_gt_arr = np.asarray(v_gt, dtype=float)

    nh = np.linalg.norm(v_hat_arr)
    ng = np.linalg.norm(v_gt_arr)

    if nh < 1e-12 or ng < 1e-12:
        return float("nan")

    cosang = np.clip(np.dot(v_hat_arr, v_gt_arr) / (nh * ng), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def score_diagnostics(score_map: np.ndarray) -> dict[str, float]:
    flat = np.asarray(score_map).ravel()
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        return {
            "top1_score": float("nan"),
            "top2_score": float("nan"),
            "peak_ratio": float("nan"),
            "peak_margin": float("nan"),
        }

    top = np.sort(flat)[::-1]
    top1 = float(top[0])
    top2 = float(top[1]) if top.size > 1 else float("nan")

    if np.isfinite(top2):
        peak_ratio = float(top1 / (top2 + 1e-12))
        peak_margin = float(top1 - top2)
    else:
        peak_ratio = float("nan")
        peak_margin = float("nan")

    return {
        "top1_score": top1,
        "top2_score": top2,
        "peak_ratio": peak_ratio,
        "peak_margin": peak_margin,
    }


# ----------------------------
# Resume helpers
# ----------------------------

def count_existing_rows(out_path: Path) -> int:
    """
    Count completed rows in an existing CSV.

    This is intentionally simple and compatible with old partial CSVs that
    do not contain all newer resume-key columns.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return 0

    try:
        return len(pd.read_csv(out_path))
    except pd.errors.EmptyDataError:
        return 0


def load_existing_results(out_path: Path) -> list[dict[str, Any]]:
    """
    Load existing rows for final summary printing.

    This does not affect resume behavior. Resume is row-count based.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return []

    try:
        df = pd.read_csv(out_path)
    except pd.errors.EmptyDataError:
        return []

    return df.to_dict(orient="records")


# ----------------------------
# GPU / worker control
# ----------------------------

def configure_worker_device(job: ExperimentJob, worker_index: int) -> None:
    """
    Assign a GPU to this process by setting CUDA_VISIBLE_DEVICES.

    This must happen before importing torch-heavy detector code.
    """
    if not job.use_gpu:
        return

    if len(job.gpu_ids) == 0:
        return

    gpu_id = job.gpu_ids[worker_index % len(job.gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# ----------------------------
# Method runners
# ----------------------------

def run_phase_prony(
    video: np.ndarray,
    job: ExperimentJob,
) -> dict[str, Any]:
    from phase_model.segmenter import FourierVisionConfig, FourierVisionSegmenter

    cfg = FourierVisionConfig(
        solver="prony",
        num_components=1,
        num_velocities=1,
        velocity_bounds=(job.velocity_min, job.velocity_max),
        velocity_bins=job.velocity_bins,
        hough_sigma=job.phase_hough_sigma,
        peak_min_separation=job.phase_peak_min_separation,
        min_frequency_radius=job.phase_min_frequency_radius,
        use_magnitude_weights=True,
    )

    detector = FourierVisionSegmenter(cfg)

    t0 = perf_counter()
    result = detector.detect(video)
    runtime = perf_counter() - t0

    return {
        "velocities": [tuple(map(float, v)) for v in result.velocities],
        "score_map": result.hough,
        "runtime_sec": runtime,
    }


def run_plane_detector(
    video: np.ndarray,
    job: ExperimentJob,
) -> dict[str, Any]:
    from plane_scoring_detection import FourierMotionConfig, FourierMotionDetector

    cfg = FourierMotionConfig(
        velocity_bounds=(job.velocity_min, job.velocity_max, job.velocity_bins),
        sigma=job.plane_sigma,
        alpha=job.plane_alpha,
        dc_bins=job.plane_dc_bins,
        keep_frac=job.plane_keep_frac,
        use_gpu=job.use_gpu,
        use_hann_window=True,
        max_detections=1,
        min_detection_separation=job.plane_min_detection_separation,
        verbose=False,
    )

    detector = FourierMotionDetector(cfg)

    t0 = perf_counter()
    result = detector.detect(video)
    runtime = perf_counter() - t0

    return {
        "velocities": [tuple(map(float, v)) for v in result.detected_velocities],
        "score_map": result.energies,
        "runtime_sec": runtime,
    }


# ----------------------------
# Row construction
# ----------------------------

def base_row_from_job(job: ExperimentJob) -> dict[str, Any]:
    """
    Fields that identify the experiment configuration.

    These fields are written for both successful and failed rows.
    """
    vx_gt, vy_gt = velocity_from_speed_direction(job.speed, job.direction_deg)

    return {
        "method": job.method,
        "shape": job.shape,
        "size": job.size,
        "T": job.T,
        "H": job.H,
        "W": job.W,
        "noise_std": job.noise_std,
        "seed": job.seed,
        "speed": job.speed,
        "direction_deg": job.direction_deg,

        "velocity_min": job.velocity_min,
        "velocity_max": job.velocity_max,
        "velocity_bins": job.velocity_bins,

        "phase_hough_sigma": job.phase_hough_sigma,
        "phase_peak_min_separation": job.phase_peak_min_separation,
        "phase_min_frequency_radius": job.phase_min_frequency_radius,

        "plane_sigma": job.plane_sigma,
        "plane_alpha": job.plane_alpha,
        "plane_dc_bins": job.plane_dc_bins,
        "plane_keep_frac": job.plane_keep_frac,
        "plane_min_detection_separation": job.plane_min_detection_separation,

        "use_gpu": job.use_gpu,
        "gpu_ids": ",".join(map(str, job.gpu_ids)),

        "vx_gt": vx_gt,
        "vy_gt": vy_gt,
    }


# ----------------------------
# Single job
# ----------------------------

def run_one_job(
    packed: tuple[int, ExperimentJob],
) -> dict[str, Any]:
    worker_index, job = packed

    try:
        configure_worker_device(job, worker_index)

        from synthetic import MovingObject, generate_synthetic_sequence

        vx_gt, vy_gt = velocity_from_speed_direction(job.speed, job.direction_deg)
        cx0, cy0 = centered_trajectory_start(job.H, job.W, vx_gt, vy_gt, job.T)

        obj = MovingObject(
            kind=job.shape,
            center=(cx0, cy0),
            velocity=(vx_gt, vy_gt),
            size=job.size,
            amplitude=1.0,
        )

        video, _, _ = generate_synthetic_sequence(
            T=job.T,
            H=job.H,
            W=job.W,
            objects=[obj],
            background=0.0,
            noise_std=job.noise_std,
            normalize=False,
            clip=False,
            seed=job.seed,
        )

        if job.method == "phase":
            out = run_phase_prony(video, job)
        elif job.method == "plane":
            out = run_plane_detector(video, job)
        else:
            raise ValueError(f"Unknown method: {job.method}")

        velocities = out["velocities"]

        if len(velocities) == 0:
            vx_hat = float("nan")
            vy_hat = float("nan")
            epe = float("inf")
            spd_err = float("inf")
            ang_err = float("nan")
        else:
            vx_hat, vy_hat = velocities[0]
            epe = endpoint_error((vx_hat, vy_hat), (vx_gt, vy_gt))
            spd_err = speed_error((vx_hat, vy_hat), (vx_gt, vy_gt))
            ang_err = angular_error_deg((vx_hat, vy_hat), (vx_gt, vy_gt))

        diag = score_diagnostics(out["score_map"])

        row = base_row_from_job(job)
        row.update(
            {
                "status": "ok",
                "error_message": "",
                "traceback": "",

                "vx_hat": vx_hat,
                "vy_hat": vy_hat,

                "endpoint_error": epe,
                "speed_error": spd_err,
                "angular_error_deg": ang_err,
                "success_at_0.1": bool(epe <= 0.1),
                "success_at_0.2": bool(epe <= 0.2),
                "success_at_0.5": bool(epe <= 0.5),

                "runtime_sec": out["runtime_sec"],
                **diag,
            }
        )
        return row

    except Exception as e:
        row = base_row_from_job(job)
        row.update(
            {
                "status": "exception",
                "error_message": str(e),
                "traceback": traceback.format_exc(),

                "vx_hat": float("nan"),
                "vy_hat": float("nan"),

                "endpoint_error": float("nan"),
                "speed_error": float("nan"),
                "angular_error_deg": float("nan"),
                "success_at_0.1": False,
                "success_at_0.2": False,
                "success_at_0.5": False,

                "runtime_sec": float("nan"),
                "top1_score": float("nan"),
                "top2_score": float("nan"),
                "peak_ratio": float("nan"),
                "peak_margin": float("nan"),
            }
        )
        return row


# ----------------------------
# Job construction
# ----------------------------

def build_jobs(args: argparse.Namespace) -> list[ExperimentJob]:
    jobs: list[ExperimentJob] = []

    gpu_ids = tuple(args.gpu_ids)

    for method, shape, size, speed, direction_deg, noise_std, T, seed in itertools.product(
        args.methods,
        args.shapes,
        args.sizes,
        args.speeds,
        args.directions,
        args.noise_levels,
        args.frame_counts,
        args.seeds,
    ):
        jobs.append(
            ExperimentJob(
                method=method,
                shape=shape,
                size=float(size),
                speed=float(speed),
                direction_deg=float(direction_deg),
                noise_std=float(noise_std),
                T=int(T),
                H=int(args.height),
                W=int(args.width),
                seed=int(seed),

                velocity_min=float(args.velocity_min),
                velocity_max=float(args.velocity_max),
                velocity_bins=int(args.velocity_bins),

                phase_hough_sigma=float(args.phase_hough_sigma),
                phase_peak_min_separation=int(args.phase_peak_min_separation),
                phase_min_frequency_radius=float(args.phase_min_frequency_radius),

                plane_sigma=float(args.plane_sigma),
                plane_alpha=float(args.plane_alpha),
                plane_dc_bins=int(args.plane_dc_bins),
                plane_keep_frac=float(args.plane_keep_frac),
                plane_min_detection_separation=int(args.plane_min_detection_separation),

                use_gpu=bool(args.use_gpu),
                gpu_ids=gpu_ids,
            )
        )

    return jobs


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="results/one_object_benchmark.csv")
    parser.add_argument("--metadata-out", type=str, default=None)

    parser.add_argument("--methods", nargs="+", default=["phase", "plane"], choices=["phase", "plane"])
    parser.add_argument("--shapes", nargs="+", default=["gaussian", "disk", "square"])
    parser.add_argument("--sizes", nargs="+", type=float, default=[5.0])

    parser.add_argument("--speeds", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument(
        "--directions",
        nargs="+",
        type=float,
        default=[0, 45, 90, 135, 180, 225, 270, 315],
    )
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)

    parser.add_argument("--velocity-min", type=float, default=-2.5)
    parser.add_argument("--velocity-max", type=float, default=2.5)
    parser.add_argument("--velocity-bins", type=int, default=151)

    # Phase knobs
    parser.add_argument("--phase-hough-sigma", type=float, default=0.05)
    parser.add_argument("--phase-peak-min-separation", type=int, default=30)
    parser.add_argument("--phase-min-frequency-radius", type=float, default=0.2)

    # Plane knobs
    parser.add_argument("--plane-sigma", type=float, default=0.3)
    parser.add_argument("--plane-alpha", type=float, default=0.05)
    parser.add_argument("--plane-dc-bins", type=int, default=2)
    parser.add_argument("--plane-keep-frac", type=float, default=0.25)
    parser.add_argument("--plane-min-detection-separation", type=int, default=8)

    # Runtime
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)

    # Resume / overwrite
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing CSV by counting completed rows and "
            "skipping that many deterministic jobs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output CSV and metadata before running.",
    )

    return parser.parse_args()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.metadata_out is None:
        metadata_path = out_path.with_suffix(".metadata.json")
    else:
        metadata_path = Path(args.metadata_out)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        if out_path.exists():
            print(f"[overwrite] Removing existing CSV: {out_path}")
            out_path.unlink()

        if metadata_path.exists():
            print(f"[overwrite] Removing existing metadata: {metadata_path}")
            metadata_path.unlink()

    jobs = build_jobs(args)

    if args.limit is not None:
        jobs = jobs[: args.limit]

    resume_n = 0
    existing_results: list[dict[str, Any]] = []

    if args.resume:
        resume_n = count_existing_rows(out_path)
        existing_results = load_existing_results(out_path)

        if resume_n > len(jobs):
            print(
                f"[resume] Existing CSV has {resume_n} rows, but current job list "
                f"has only {len(jobs)} jobs."
            )
            print(
                "[resume] This usually means you changed CLI arguments. "
                "Refusing to continue because row-count resume would be unsafe."
            )
            raise ValueError("Existing CSV has more rows than current job list.")

        print(f"[resume] Existing rows found: {resume_n}")
        print(f"[resume] These rows will be kept.")
    elif out_path.exists() and out_path.stat().st_size > 0:
        print(f"[warning] Output CSV already exists: {out_path}")
        print("[warning] Not using --resume and not using --overwrite.")
        print("[warning] New rows will be appended to the existing CSV.")
        print("[warning] Usually you want either --resume or --overwrite.")

    jobs_to_run = jobs[resume_n:]

    print(f"Total jobs in full grid: {len(jobs)}")
    print(f"Jobs already completed: {resume_n}")
    print(f"Jobs remaining: {len(jobs_to_run)}")
    print(f"Methods: {args.methods}")
    print(f"Shapes: {args.shapes}")
    print(f"Output: {out_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Workers: {args.num_workers}")
    print(f"Use GPU: {args.use_gpu}")
    if args.use_gpu:
        print(f"GPU IDs: {args.gpu_ids}")

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "num_jobs_full_grid": len(jobs),
                "num_jobs_completed_before_this_run": resume_n,
                "num_jobs_remaining_at_start": len(jobs_to_run),
                "jobs_example": asdict(jobs[0]) if len(jobs) > 0 else None,
            },
            f,
            indent=2,
        )

    if len(jobs_to_run) == 0:
        print("\nNothing to do. All jobs are already completed.")
        return

    # Keep old results only for summary counts. We do not rewrite the old CSV.
    results: list[dict[str, Any]] = list(existing_results)

    t_global = perf_counter()

    # Important:
    # The worker_index passed here is local to this resumed run, not the original
    # full-grid index. That is fine because it is only used to assign GPUs.
    packed_jobs = list(enumerate(jobs_to_run))

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(run_one_job, packed)
            for packed in packed_jobs
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)

            df_one = pd.DataFrame([row])
            write_header = not out_path.exists() or out_path.stat().st_size == 0
            df_one.to_csv(out_path, mode="a", header=write_header, index=False)

            if i % 2 == 0 or i == len(jobs_to_run):
                elapsed = perf_counter() - t_global
                ok = sum(r.get("status") == "ok" for r in results)
                fail = len(results) - ok

                print(
                    f"{i}/{len(jobs_to_run)} newly done | "
                    f"total_rows_now={len(results)} | "
                    f"ok={ok} fail={fail} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    df = pd.DataFrame(results)

    print("\nDone.")
    print(f"Saved: {out_path}")
    print(f"Saved metadata: {metadata_path}")

    if len(df) > 0:
        print("\nStatus counts:")
        print(df["status"].value_counts(dropna=False))

        if "endpoint_error" in df.columns:
            print("\nTop-level summary:")
            summary = (
                df[df["status"] == "ok"]
                .groupby("method")
                .agg(
                    mean_epe=("endpoint_error", "mean"),
                    median_epe=("endpoint_error", "median"),
                    success_0_2=("success_at_0.2", "mean"),
                    mean_runtime=("runtime_sec", "mean"),
                    n=("endpoint_error", "count"),
                )
            )
            print(summary)


if __name__ == "__main__":
    main()