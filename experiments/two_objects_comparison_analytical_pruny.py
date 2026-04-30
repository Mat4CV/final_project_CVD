"""
Run two-object Fourier phase motion benchmarks.

Compares:
    1. Analytical phase detector from phase_model / fourier_vision
    2. Prony phase detector from phase_model / fourier_vision

This experiment tests whether phase-based Fourier methods can recover
two simultaneous translational velocities from one synthetic video.

Example smoke test:

    uv run python experiments/two_object_phase_comparison.py \
        --out results/two_object_phase_smoke.csv \
        --methods analytic prony \
        --shapes gaussian \
        --speeds1 0.75 \
        --speeds2 1.25 \
        --directions1 0 45 \
        --directions2 180 225 \
        --noise-levels 0.0 \
        --frame-counts 32 \
        --seeds 0 \
        --height 128 \
        --width 128 \
        --limit 8 \
        --overwrite

Limited benchmark:

    uv run python experiments/two_object_phase_comparison.py \
        --out results/two_object_phase_small.csv \
        --methods analytic prony \
        --shapes gaussian disk \
        --speeds1 0.5 1.0 \
        --speeds2 0.5 1.0 \
        --directions1 0 45 90 135 \
        --directions2 180 225 270 315 \
        --noise-levels 0.0 0.05 \
        --frame-counts 32 64 \
        --seeds 0 1 \
        --height 128 \
        --width 128 \
        --velocity-min -2.5 \
        --velocity-max 2.5 \
        --velocity-bins 151 \
        --num-workers 4 \
        --resume

Notes:
    - This script is intentionally similar to one_object_comparison.py.
    - Resume is row-count based.
    - The two predicted velocities are matched to the two GT velocities
      by minimum total endpoint error.
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

    size1: float
    size2: float
    amplitude1: float
    amplitude2: float
    object_separation: float

    speed1: float
    direction1_deg: float
    speed2: float
    direction2_deg: float

    noise_std: float
    T: int
    H: int
    W: int
    seed: int

    velocity_min: float
    velocity_max: float
    velocity_bins: int
    min_velocity_separation: float

    # Phase method knobs
    phase_hough_sigma: float
    phase_peak_min_separation: int
    phase_min_frequency_radius: float

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
    Choose initial center so the full trajectory is centered in the image.

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


def score_diagnostics(score_map: np.ndarray | None) -> dict[str, float]:
    if score_map is None:
        return {
            "top1_score": float("nan"),
            "top2_score": float("nan"),
            "peak_ratio": float("nan"),
            "peak_margin": float("nan"),
        }

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


def match_two_velocities(
    preds: list[tuple[float, float]],
    gts: list[tuple[float, float]],
) -> dict[str, float]:
    """
    Match up to two predicted velocities to two ground-truth velocities.

    Missing predictions receive infinite error.

    Returns predicted velocities in GT order:
        vx1_hat, vy1_hat corresponds to GT object 1
        vx2_hat, vy2_hat corresponds to GT object 2
    """
    if len(gts) != 2:
        raise ValueError("This helper assumes exactly two ground-truth velocities.")

    if len(preds) == 0:
        return {
            "vx1_hat": float("nan"),
            "vy1_hat": float("nan"),
            "vx2_hat": float("nan"),
            "vy2_hat": float("nan"),
            "matched_epe1": float("inf"),
            "matched_epe2": float("inf"),
            "mean_endpoint_error": float("inf"),
            "max_endpoint_error": float("inf"),
            "speed_error1": float("inf"),
            "speed_error2": float("inf"),
            "angular_error1_deg": float("nan"),
            "angular_error2_deg": float("nan"),
        }

    if len(preds) == 1:
        p = np.asarray(preds[0], dtype=float)
        g0 = np.asarray(gts[0], dtype=float)
        g1 = np.asarray(gts[1], dtype=float)

        e0 = float(np.linalg.norm(p - g0))
        e1 = float(np.linalg.norm(p - g1))

        if e0 <= e1:
            vx1_hat, vy1_hat = preds[0]
            vx2_hat, vy2_hat = float("nan"), float("nan")
            epe1, epe2 = e0, float("inf")
            spd1 = speed_error(preds[0], gts[0])
            spd2 = float("inf")
            ang1 = angular_error_deg(preds[0], gts[0])
            ang2 = float("nan")
        else:
            vx1_hat, vy1_hat = float("nan"), float("nan")
            vx2_hat, vy2_hat = preds[0]
            epe1, epe2 = float("inf"), e1
            spd1 = float("inf")
            spd2 = speed_error(preds[0], gts[1])
            ang1 = float("nan")
            ang2 = angular_error_deg(preds[0], gts[1])

        return {
            "vx1_hat": float(vx1_hat),
            "vy1_hat": float(vy1_hat),
            "vx2_hat": float(vx2_hat),
            "vy2_hat": float(vy2_hat),
            "matched_epe1": epe1,
            "matched_epe2": epe2,
            "mean_endpoint_error": float("inf"),
            "max_endpoint_error": float("inf"),
            "speed_error1": spd1,
            "speed_error2": spd2,
            "angular_error1_deg": ang1,
            "angular_error2_deg": ang2,
        }

    # Only use the first two predicted velocities for the main benchmark.
    # Usually these should be ordered by score by the detector.
    preds = preds[:2]

    p0 = np.asarray(preds[0], dtype=float)
    p1 = np.asarray(preds[1], dtype=float)
    g0 = np.asarray(gts[0], dtype=float)
    g1 = np.asarray(gts[1], dtype=float)

    cost_same = float(np.linalg.norm(p0 - g0) + np.linalg.norm(p1 - g1))
    cost_swap = float(np.linalg.norm(p0 - g1) + np.linalg.norm(p1 - g0))

    if cost_same <= cost_swap:
        pred1 = preds[0]
        pred2 = preds[1]
    else:
        pred1 = preds[1]
        pred2 = preds[0]

    epe1 = endpoint_error(pred1, gts[0])
    epe2 = endpoint_error(pred2, gts[1])

    return {
        "vx1_hat": float(pred1[0]),
        "vy1_hat": float(pred1[1]),
        "vx2_hat": float(pred2[0]),
        "vy2_hat": float(pred2[1]),
        "matched_epe1": epe1,
        "matched_epe2": epe2,
        "mean_endpoint_error": float(0.5 * (epe1 + epe2)),
        "max_endpoint_error": float(max(epe1, epe2)),
        "speed_error1": speed_error(pred1, gts[0]),
        "speed_error2": speed_error(pred2, gts[1]),
        "angular_error1_deg": angular_error_deg(pred1, gts[0]),
        "angular_error2_deg": angular_error_deg(pred2, gts[1]),
    }


# ----------------------------
# Resume helpers
# ----------------------------

def count_existing_rows(out_path: Path) -> int:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return 0

    try:
        return len(pd.read_csv(out_path))
    except pd.errors.EmptyDataError:
        return 0


def load_existing_results(out_path: Path) -> list[dict[str, Any]]:
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
    if not job.use_gpu:
        return

    if len(job.gpu_ids) == 0:
        return

    gpu_id = job.gpu_ids[worker_index % len(job.gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# ----------------------------
# Method runners
# ----------------------------

def run_phase_method(
    video: np.ndarray,
    job: ExperimentJob,
) -> dict[str, Any]:
    """
    Run analytical or Prony phase detector.

    This assumes your FourierVisionConfig supports:
        solver="closed_form_m2"
        solver="prony"
    """
    from phase_model.segmenter import FourierVisionConfig, FourierVisionSegmenter

    if job.method == "closed_form_m2":
        solver_name = "closed_form_m2"
    elif job.method == "prony":
        solver_name = "prony"
    else:
        raise ValueError(f"Unknown phase method: {job.method}")

    cfg = FourierVisionConfig(
        solver=solver_name,
        num_components=2,
        num_velocities=2,
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

    velocities = [tuple(map(float, v)) for v in result.velocities]

    score_map = None
    if hasattr(result, "hough"):
        score_map = result.hough

    return {
        "velocities": velocities,
        "score_map": score_map,
        "runtime_sec": runtime,
    }


# ----------------------------
# Row construction
# ----------------------------

def base_row_from_job(job: ExperimentJob) -> dict[str, Any]:
    vx1_gt, vy1_gt = velocity_from_speed_direction(job.speed1, job.direction1_deg)
    vx2_gt, vy2_gt = velocity_from_speed_direction(job.speed2, job.direction2_deg)

    return {
        "method": job.method,
        "shape": job.shape,
        "size1": job.size1,
        "size2": job.size2,
        "amplitude1": job.amplitude1,
        "amplitude2": job.amplitude2,
        "object_separation": job.object_separation,
        "T": job.T,
        "H": job.H,
        "W": job.W,
        "noise_std": job.noise_std,
        "seed": job.seed,

        "speed1": job.speed1,
        "direction1_deg": job.direction1_deg,
        "vx1_gt": vx1_gt,
        "vy1_gt": vy1_gt,

        "speed2": job.speed2,
        "direction2_deg": job.direction2_deg,
        "vx2_gt": vx2_gt,
        "vy2_gt": vy2_gt,

        "velocity_min": job.velocity_min,
        "velocity_max": job.velocity_max,
        "velocity_bins": job.velocity_bins,
        "min_velocity_separation": job.min_velocity_separation,

        "phase_hough_sigma": job.phase_hough_sigma,
        "phase_peak_min_separation": job.phase_peak_min_separation,
        "phase_min_frequency_radius": job.phase_min_frequency_radius,

        "use_gpu": job.use_gpu,
        "gpu_ids": ",".join(map(str, job.gpu_ids)),
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

        vx1_gt, vy1_gt = velocity_from_speed_direction(job.speed1, job.direction1_deg)
        vx2_gt, vy2_gt = velocity_from_speed_direction(job.speed2, job.direction2_deg)

        cx1, cy1 = centered_trajectory_start(job.H, job.W, vx1_gt, vy1_gt, job.T)
        cx2, cy2 = centered_trajectory_start(job.H, job.W, vx2_gt, vy2_gt, job.T)

        # Separate the two trajectories in image space.
        # Object 1 moves around left side, object 2 around right side.
        cx1 -= 0.5 * job.object_separation
        cx2 += 0.5 * job.object_separation

        obj1 = MovingObject(
            kind=job.shape,
            center=(cx1, cy1),
            velocity=(vx1_gt, vy1_gt),
            size=job.size1,
            amplitude=job.amplitude1,
        )

        obj2 = MovingObject(
            kind=job.shape,
            center=(cx2, cy2),
            velocity=(vx2_gt, vy2_gt),
            size=job.size2,
            amplitude=job.amplitude2,
        )

        video, _, _ = generate_synthetic_sequence(
            T=job.T,
            H=job.H,
            W=job.W,
            objects=[obj1, obj2],
            background=0.0,
            noise_std=job.noise_std,
            normalize=False,
            clip=False,
            seed=job.seed,
        )

        out = run_phase_method(video, job)

        velocities = out["velocities"]
        gts = [(vx1_gt, vy1_gt), (vx2_gt, vy2_gt)]

        match = match_two_velocities(
            preds=velocities,
            gts=gts,
        )

        diag = score_diagnostics(out["score_map"])

        epe1 = match["matched_epe1"]
        epe2 = match["matched_epe2"]

        row = base_row_from_job(job)
        row.update(
            {
                "status": "ok",
                "error_message": "",
                "traceback": "",

                "num_detected": int(len(velocities)),

                **match,

                "success_any_at_0.1": bool(epe1 <= 0.1 or epe2 <= 0.1),
                "success_any_at_0.2": bool(epe1 <= 0.2 or epe2 <= 0.2),
                "success_any_at_0.5": bool(epe1 <= 0.5 or epe2 <= 0.5),

                "success_both_at_0.1": bool(epe1 <= 0.1 and epe2 <= 0.1),
                "success_both_at_0.2": bool(epe1 <= 0.2 and epe2 <= 0.2),
                "success_both_at_0.5": bool(epe1 <= 0.5 and epe2 <= 0.5),

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

                "num_detected": 0,

                "vx1_hat": float("nan"),
                "vy1_hat": float("nan"),
                "vx2_hat": float("nan"),
                "vy2_hat": float("nan"),

                "matched_epe1": float("nan"),
                "matched_epe2": float("nan"),
                "mean_endpoint_error": float("nan"),
                "max_endpoint_error": float("nan"),
                "speed_error1": float("nan"),
                "speed_error2": float("nan"),
                "angular_error1_deg": float("nan"),
                "angular_error2_deg": float("nan"),

                "success_any_at_0.1": False,
                "success_any_at_0.2": False,
                "success_any_at_0.5": False,
                "success_both_at_0.1": False,
                "success_both_at_0.2": False,
                "success_both_at_0.5": False,

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

    for (
        method,
        shape,
        size1,
        size2,
        speed1,
        speed2,
        direction1_deg,
        direction2_deg,
        noise_std,
        T,
        seed,
    ) in itertools.product(
        args.methods,
        args.shapes,
        args.sizes1,
        args.sizes2,
        args.speeds1,
        args.speeds2,
        args.directions1,
        args.directions2,
        args.noise_levels,
        args.frame_counts,
        args.seeds,
    ):
        vx1, vy1 = velocity_from_speed_direction(speed1, direction1_deg)
        vx2, vy2 = velocity_from_speed_direction(speed2, direction2_deg)

        vel_sep = float(
            np.linalg.norm(
                np.asarray([vx1, vy1], dtype=float)
                - np.asarray([vx2, vy2], dtype=float)
            )
        )

        if vel_sep < args.min_velocity_separation:
            continue

        jobs.append(
            ExperimentJob(
                method=str(method),
                shape=str(shape),

                size1=float(size1),
                size2=float(size2),
                amplitude1=float(args.amplitude1),
                amplitude2=float(args.amplitude2),
                object_separation=float(args.object_separation),

                speed1=float(speed1),
                direction1_deg=float(direction1_deg),
                speed2=float(speed2),
                direction2_deg=float(direction2_deg),

                noise_std=float(noise_std),
                T=int(T),
                H=int(args.height),
                W=int(args.width),
                seed=int(seed),

                velocity_min=float(args.velocity_min),
                velocity_max=float(args.velocity_max),
                velocity_bins=int(args.velocity_bins),
                min_velocity_separation=float(args.min_velocity_separation),

                phase_hough_sigma=float(args.phase_hough_sigma),
                phase_peak_min_separation=int(args.phase_peak_min_separation),
                phase_min_frequency_radius=float(args.phase_min_frequency_radius),

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

    parser.add_argument("--out", type=str, default="results/two_object_phase_benchmark.csv")
    parser.add_argument("--metadata-out", type=str, default=None)

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["closed_form_m2", "prony"],
        choices=["closed_form_m2", "prony"],
    )

    parser.add_argument("--shapes", nargs="+", default=["gaussian", "disk", "square"])

    parser.add_argument("--sizes1", nargs="+", type=float, default=[5.0])
    parser.add_argument("--sizes2", nargs="+", type=float, default=[5.0])

    parser.add_argument("--amplitude1", type=float, default=1.0)
    parser.add_argument("--amplitude2", type=float, default=1.0)
    parser.add_argument("--object-separation", type=float, default=36.0)

    parser.add_argument("--speeds1", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--speeds2", nargs="+", type=float, default=[0.5, 1.0])

    parser.add_argument(
        "--directions1",
        nargs="+",
        type=float,
        default=[0, 45, 90, 135],
    )
    parser.add_argument(
        "--directions2",
        nargs="+",
        type=float,
        default=[180, 225, 270, 315],
    )

    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.05])
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])

    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)

    parser.add_argument("--velocity-min", type=float, default=-2.5)
    parser.add_argument("--velocity-max", type=float, default=2.5)
    parser.add_argument("--velocity-bins", type=int, default=151)
    parser.add_argument("--min-velocity-separation", type=float, default=0.5)

    # Phase knobs
    parser.add_argument("--phase-hough-sigma", type=float, default=0.05)
    parser.add_argument("--phase-peak-min-separation", type=int, default=30)
    parser.add_argument("--phase-min-frequency-radius", type=float, default=0.2)

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
        print("[resume] These rows will be kept.")
    elif out_path.exists() and out_path.stat().st_size > 0:
        print(f"[warning] Output CSV already exists: {out_path}")
        print("[warning] Not using --resume and not using --overwrite.")
        print("[warning] New rows will be appended to the existing CSV.")
        print("[warning] Usually you want either --resume or --overwrite.")

    jobs_to_run = jobs[resume_n:]

    print(f"Total jobs in full grid after velocity-separation filtering: {len(jobs)}")
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
                "num_jobs_full_grid_after_filtering": len(jobs),
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

    results: list[dict[str, Any]] = list(existing_results)

    t_global = perf_counter()

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

        if "mean_endpoint_error" in df.columns:
            print("\nTop-level summary:")
            summary = (
                df[df["status"] == "ok"]
                .groupby("method")
                .agg(
                    mean_epe=("mean_endpoint_error", "mean"),
                    median_epe=("mean_endpoint_error", "median"),
                    max_epe=("max_endpoint_error", "mean"),
                    success_both_0_2=("success_both_at_0.2", "mean"),
                    success_any_0_2=("success_any_at_0.2", "mean"),
                    mean_num_detected=("num_detected", "mean"),
                    mean_runtime=("runtime_sec", "mean"),
                    n=("mean_endpoint_error", "count"),
                )
            )
            print(summary)


if __name__ == "__main__":
    main()