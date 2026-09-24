#!/usr/bin/env python3
"""
Batch-run the dataset pipeline for 31 July .bag files.

For each rosbag this script runs the steps:
 1. 1_rosbag_to_csv_vrpn.py
 2. 3_csv_dataset_cleaner.py
 3. 4_align_gt_and_imu.py
 4. 2_plot_csv_data.py (optional, for plots)

It writes the final cleaned, time-aligned dataset as:

    <out_root>/<optional RAW or UN>/flight_n/imu.csv
    <out_root>/<optional RAW or UN>/flight_n/data.csv
"""

from pathlib import Path
from collections import Counter
import subprocess
import sys
import shutil
import argparse
import re


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = WORKSPACE_ROOT / "scripts" / "pipeline"

import numpy as np
import pandas as pd


IMU_COLUMNS = [
    "time",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
]

DATA_COLUMNS = [
    "time",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
]


def run_command(cmd, cwd=None):
    print("Running:", " ".join(str(p) for p in cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    result.check_returncode()


def natural_flight_key(path: Path):
    match = re.search(r"flight_(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return path.stem


def _load_final_csv(path: Path, columns):
    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    df = df[columns].copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return df


def write_canonical_dataset(imu_path: Path, data_path: Path, out_dir: Path) -> None:
    imu = _load_final_csv(imu_path, IMU_COLUMNS)
    data = _load_final_csv(data_path, DATA_COLUMNS)

    if len(imu) != len(data) or not np.allclose(
        imu["time"].to_numpy(),
        data["time"].to_numpy(),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Final IMU and data timelines are not aligned: "
            f"{imu_path} rows={len(imu)}, {data_path} rows={len(data)}"
        )

    valid_rows = (
        np.isfinite(imu[IMU_COLUMNS].to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(data[DATA_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    )
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows:
        imu = imu.loc[valid_rows].reset_index(drop=True)
        data = data.loc[valid_rows].reset_index(drop=True)
        print(f"Dropped {dropped_rows} row(s) with NaN/inf values before final export")

    if imu.empty or data.empty:
        raise ValueError(f"No valid aligned rows left for {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    imu.to_csv(out_dir / "imu.csv", index=False)
    data.to_csv(out_dir / "data.csv", index=False)
    print(f"Wrote final dataset -> {out_dir / 'imu.csv'}")
    print(f"Wrote final dataset -> {out_dir / 'data.csv'}")


def process_bag(
    bag_path: Path,
    out_dir: Path,
    scripts_dir: Path,
    do_plot: bool = True,
    keep_intermediate: bool = True,
    clean_bounds=None,
) -> bool:
    bag = bag_path
    BASE = bag.with_suffix("")
    IMU_CSV = Path(f"{BASE}_imu.csv")
    GT_CSV = Path(f"{BASE}_gt.csv")
    IMU_CLEAN_CSV = Path(f"{BASE}_imu_clean.csv")
    GT_CLEAN_CSV = Path(f"{BASE}_gt_clean.csv")
    IMU_200HZ_CSV = Path(f"{BASE}_imu_clean_200hz.csv")
    GT_ALIGNED_CSV = Path(f"{BASE}_gt_clean_aligned.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: rosbag -> raw IMU and GT CSV
        run_command([
            sys.executable,
            "1_rosbag_to_csv_vrpn.py",
            "--bag",
            str(bag),
            "--imu_out",
            str(IMU_CSV),
            "--gt_out",
            str(GT_CSV),
        ], cwd=str(scripts_dir))

        # Step 3: clean CSVs
        clean_cmd = [
            sys.executable,
            "3_csv_dataset_cleaner.py",
            "--imu_csv",
            str(IMU_CSV),
            "--gt_csv",
            str(GT_CSV),
            "--savgol_window",
            "21",
            "--savgol_polyorder",
            "2",
        ]
        for bound in clean_bounds or []:
            clean_cmd.extend(["--bound", bound])
        run_command(clean_cmd, cwd=str(scripts_dir))

        # Step 4: align GT to IMU timeline and create 200 Hz IMU
        run_command([
            sys.executable,
            "4_align_gt_and_imu.py",
            "--imu_csv",
            str(IMU_CLEAN_CSV),
            "--gt_csv",
            str(GT_CLEAN_CSV),
            "--resample_imu_hz",
            "200",
            "--max_interp_gap",
            "0.05",
            "--imu_resampled_out",
            str(IMU_200HZ_CSV),
            "--out",
            str(GT_ALIGNED_CSV),
        ], cwd=str(scripts_dir))

        # Optional Step 2: produce plots (kept for completeness)
        if do_plot:
            plots_dir = out_dir / "plots"
            run_command([
                sys.executable,
                "2_plot_csv_data.py",
                "--imu_csv",
                str(IMU_200HZ_CSV),
                "--gt_csv",
                str(GT_ALIGNED_CSV),
                "--out_dir",
                str(plots_dir),
            ], cwd=str(scripts_dir))

        write_canonical_dataset(IMU_200HZ_CSV, GT_ALIGNED_CSV, out_dir)

        if keep_intermediate:
            shutil.copy2(IMU_200HZ_CSV, out_dir / IMU_200HZ_CSV.name)
            shutil.copy2(GT_ALIGNED_CSV, out_dir / GT_ALIGNED_CSV.name)

        return True

    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error processing {bag}: {e}", file=sys.stderr)
        return False


def find_bags(search_dir: Path):
    return sorted(
        search_dir.rglob("flight_*.bag"),
        key=lambda path: (path.parent.as_posix(), natural_flight_key(path)),
    )


def output_dir_for_bag(
    bag: Path,
    input_dir: Path,
    out_root: Path,
    split_by_parent: str,
    has_multiple_parent_dirs: bool,
    output_name: str = None,
) -> Path:
    rel = bag.relative_to(input_dir)
    should_split = split_by_parent == "always" or (
        split_by_parent == "auto" and has_multiple_parent_dirs
    )

    if should_split and len(rel.parts) > 1:
        return out_root / rel.parts[0] / (output_name or bag.stem)

    return out_root / (output_name or bag.stem)


def main():
    p = argparse.ArgumentParser(description="Run dataset pipeline for 31 July rosbags")
    p.add_argument("--input_dir", default=WORKSPACE_ROOT / "data" / "rosbags" / "2026-07-31", help="Directory to search for flight_*.bag files")
    p.add_argument("--out_root", default=WORKSPACE_ROOT / "results" / "31st_July_dataset", help="Root folder to store per-flight outputs")
    p.add_argument("--scripts_dir", default=PIPELINE_DIR, help="Directory containing the 1..4 scripts")
    p.add_argument("--no_plots", action="store_true", help="Skip plotting step (Step 2)")
    p.add_argument(
        "--split_by_parent",
        choices=["auto", "always", "never"],
        default="auto",
        help="Place outputs under parent folder names such as RAW/ and UN/",
    )
    p.add_argument(
        "--no_intermediate_copies",
        action="store_true",
        help="Only keep imu.csv and data.csv in each flight folder",
    )
    p.add_argument(
        "--clean_bound",
        action="append",
        default=[],
        help="Forward a cleaner bound override as pattern:min:max, e.g. vel_*:-4:4",
    )
    p.add_argument(
        "--sequential_names",
        action="store_true",
        help="Name output folders flight_1, flight_2, ... within each parent split",
    )
    p.add_argument("--dry_run", action="store_true", help="Print planned input -> output mapping and exit")
    args = p.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_root = Path(args.out_root).resolve()
    scripts_dir = Path(args.scripts_dir).resolve()

    bags = find_bags(input_dir)
    if not bags:
        print(f"No .bag files found under {input_dir}")
        return

    duplicate_stems = Counter(bag.stem for bag in bags)
    parent_dirs = {
        bag.relative_to(input_dir).parts[0]
        for bag in bags
        if len(bag.relative_to(input_dir).parts) > 1
    }
    has_multiple_parent_dirs = len(parent_dirs) > 1
    if args.split_by_parent == "never":
        duplicates = sorted(stem for stem, count in duplicate_stems.items() if count > 1)
        if duplicates:
            raise SystemExit(
                "Duplicate flight names would overwrite each other. "
                f"Use --split_by_parent auto/always or process one folder at a time: {duplicates}"
            )

    print(f"Found {len(bags)} bag(s) to process")
    sequence_names = {}
    if args.sequential_names:
        counters = Counter()
        for bag in bags:
            rel = bag.relative_to(input_dir)
            sequence_key = rel.parts[0] if len(rel.parts) > 1 else ""
            counters[sequence_key] += 1
            sequence_names[bag] = f"flight_{counters[sequence_key]}"

    if args.dry_run:
        for bag in bags:
            out_dir = output_dir_for_bag(
                bag,
                input_dir,
                out_root,
                args.split_by_parent,
                has_multiple_parent_dirs,
                output_name=sequence_names.get(bag),
            )
            print(f"{bag} -> {out_dir / 'imu.csv'} and {out_dir / 'data.csv'}")
        return

    failures = []
    for bag in bags:
        print("\n=== Processing", bag, "===")
        out_dir = output_dir_for_bag(
            bag,
            input_dir,
            out_root,
            args.split_by_parent,
            has_multiple_parent_dirs,
            output_name=sequence_names.get(bag),
        )
        ok = process_bag(
            bag,
            out_dir,
            scripts_dir,
            do_plot=not args.no_plots,
            keep_intermediate=not args.no_intermediate_copies,
            clean_bounds=args.clean_bound,
        )
        if not ok:
            failures.append(bag)

    if failures:
        print("\nFailed bag(s):", file=sys.stderr)
        for bag in failures:
            print(f"  {bag}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
