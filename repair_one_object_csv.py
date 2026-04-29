from pathlib import Path
import csv
import shutil
import pandas as pd

path = Path("results/one_object/one_object_square.csv")

backup = path.with_suffix(".broken.csv")
fixed = path.with_suffix(".fixed.csv")

shutil.copy2(path, backup)
print(f"Backup saved to: {backup}")

TARGET_COLUMNS = [
    "status",
    "error_message",
    "method",
    "shape",
    "size",
    "T",
    "H",
    "W",
    "noise_std",
    "seed",
    "speed",
    "direction_deg",
    "vx_gt",
    "vy_gt",
    "vx_hat",
    "vy_hat",
    "endpoint_error",
    "speed_error",
    "angular_error_deg",
    "success_at_0.1",
    "success_at_0.2",
    "success_at_0.5",
    "runtime_sec",
    "top1_score",
    "top2_score",
    "peak_ratio",
    "peak_margin",
]

NEW_COLUMNS = [
    "method",
    "shape",
    "size",
    "T",
    "H",
    "W",
    "noise_std",
    "seed",
    "speed",
    "direction_deg",

    "velocity_min",
    "velocity_max",
    "velocity_bins",

    "phase_hough_sigma",
    "phase_peak_min_separation",
    "phase_min_frequency_radius",

    "plane_sigma",
    "plane_alpha",
    "plane_dc_bins",
    "plane_keep_frac",
    "plane_min_detection_separation",

    "use_gpu",
    "gpu_ids",

    "vx_gt",
    "vy_gt",

    "status",
    "error_message",
    "traceback",

    "vx_hat",
    "vy_hat",
    "endpoint_error",
    "speed_error",
    "angular_error_deg",
    "success_at_0.1",
    "success_at_0.2",
    "success_at_0.5",
    "runtime_sec",
    "top1_score",
    "top2_score",
    "peak_ratio",
    "peak_margin",
]

rows_fixed = []
bad_rows = []

with open(path, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    for line_num, row in enumerate(reader, start=2):
        if len(row) == 0 or all(x.strip() == "" for x in row):
            continue

        # Old/simple schema: already matches TARGET_COLUMNS
        if len(row) == len(TARGET_COLUMNS):
            d = dict(zip(TARGET_COLUMNS, row))
            rows_fixed.append({col: d.get(col, "") for col in TARGET_COLUMNS})

        # New/full schema from resumed script
        elif len(row) == len(NEW_COLUMNS):
            d = dict(zip(NEW_COLUMNS, row))
            rows_fixed.append({col: d.get(col, "") for col in TARGET_COLUMNS})

        else:
            bad_rows.append((line_num, len(row), row[:8]))

print(f"Fixed rows: {len(rows_fixed)}")
print(f"Bad rows: {len(bad_rows)}")

if bad_rows:
    print("Rows with unexpected length:")
    for item in bad_rows[:20]:
        print(item)
    raise RuntimeError("Some rows could not be repaired automatically.")

df = pd.DataFrame(rows_fixed, columns=TARGET_COLUMNS)

df.to_csv(fixed, index=False)
print(f"Fixed CSV saved to: {fixed}")

print()
print("Status counts:")
print(df["status"].value_counts(dropna=False))

print()
print("Rows by method:")
print(df["method"].value_counts(dropna=False))

print()
print("Rows by shape/method:")
print(df.groupby(["shape", "method"]).size())