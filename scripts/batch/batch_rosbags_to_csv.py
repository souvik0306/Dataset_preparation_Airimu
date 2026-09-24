#!/usr/bin/env python3
"""Convert one or more ROS 1 bag groups into clean, aligned CSV datasets.

Output layout:

    <output-root>/<motion>/<source>/flight_<1..n>/imu.csv
    <output-root>/<motion>/<source>/flight_<1..n>/data.csv

For example:

    csv_datasets/Hover/AI/flight_1/imu.csv
    csv_datasets/Hover/AI/flight_1/data.csv

The conversion uses the existing extraction, cleaning, and alignment scripts in
this repository. Intermediate files are created in a temporary directory, so
the rosbag directories are not polluted with generated CSV files. Each flight
also receives a ``plots/`` directory generated from its final CSV files.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "pipeline"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE_ROOT / "data" / "rosbags"

# These paths make the common six-group conversion a single command:
#     python3 batch_rosbags_to_csv.py --all-known
KNOWN_GROUPS = {
    "Low-Dynamic": DATA_ROOT / "9th_September_Low_Dynamic_Rosbags",
    "Medium-Dynamic": DATA_ROOT / "9sept_medium_dynamic_Rosbags",
    "High-Dynamic": DATA_ROOT / "10th_September_High_Dyn_Rosbags",
    "Super-High-Dynamic": DATA_ROOT / "10th_September_Super_High_Dyn",
    "Hover": DATA_ROOT / "9th_September_Hover_Rosbags",
    "Yaw": DATA_ROOT / "10th_September_Yaw_Turn",
}

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


@dataclass(frozen=True)
class BagJob:
    group: str
    input_root: Path
    bag: Path
    flight_number: int
    output_dir: Path


def slug(value: str) -> str:
    """Return a filesystem-friendly label while keeping names readable."""
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    result = result.strip("._-")
    if not result:
        raise ValueError(f"Cannot make a group name from {value!r}")
    return result


def infer_group(path: Path) -> str:
    name = path.name.lower()
    if "hover" in name:
        return "Hover"
    if "yaw" in name:
        return "Yaw"
    if "super" in name and "high" in name:
        return "Super-High-Dynamic"
    if "low" in name:
        return "Low-Dynamic"
    if "medium" in name:
        return "Medium-Dynamic"
    if "high" in name:
        return "High-Dynamic"
    return slug(path.name)


def parse_group_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid group {spec!r}; expected NAME=/path/to/rosbags"
        )
    name, raw_path = spec.split("=", 1)
    try:
        label = slug(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not raw_path.strip():
        raise argparse.ArgumentTypeError(f"Missing path in group {spec!r}")
    return label, Path(raw_path).expanduser()


def natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.as_posix().lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def find_bags(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".bag" else []
    return sorted(root.rglob("*.bag"), key=natural_key)


def output_dir_for_bag(
    output_root: Path,
    group: str,
    input_root: Path,
    bag: Path,
    flight_number: int,
) -> Path:
    if input_root.is_file():
        relative_parent = Path()
    else:
        relative_parent = bag.relative_to(input_root).parent

    # Preserve folders such as AI/ and RAW/ and any deeper source grouping.
    clean_parent = Path(*(slug(part) for part in relative_parent.parts))
    return output_root / slug(group) / clean_parent / f"flight_{flight_number}"


def build_jobs(
    groups: Iterable[tuple[str, Path]], output_root: Path
) -> list[BagJob]:
    jobs: list[BagJob] = []
    seen_outputs: dict[Path, Path] = {}

    for group, unresolved_root in groups:
        root = unresolved_root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input does not exist: {root}")
        bags = find_bags(root)
        if not bags:
            raise FileNotFoundError(f"No .bag files found under: {root}")

        flight_counts: dict[Path, int] = {}
        for bag in bags:
            relative_parent = (
                Path() if root.is_file() else bag.relative_to(root).parent
            )
            flight_counts[relative_parent] = flight_counts.get(relative_parent, 0) + 1
            flight_number = flight_counts[relative_parent]
            out_dir = output_dir_for_bag(
                output_root,
                group,
                root,
                bag,
                flight_number,
            )
            previous = seen_outputs.get(out_dir)
            if previous is not None and previous != bag:
                raise ValueError(
                    f"Output collision: {previous} and {bag} both map to {out_dir}"
                )
            seen_outputs[out_dir] = bag
            jobs.append(BagJob(group, root, bag, flight_number, out_dir))

    return sorted(jobs, key=lambda job: natural_key(job.output_dir))


def run_checked(command: Sequence[object]) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"    $ {printable}")
    subprocess.run([str(item) for item in command], check=True)


def load_numeric_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    frame = frame[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time"])
    frame = frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return frame


def write_dataset(imu_source: Path, data_source: Path, output_dir: Path) -> int:
    imu = load_numeric_csv(imu_source, IMU_COLUMNS)
    data = load_numeric_csv(data_source, DATA_COLUMNS)

    if len(imu) != len(data) or not np.allclose(
        imu["time"].to_numpy(),
        data["time"].to_numpy(),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"IMU and ground-truth timelines are not aligned "
            f"(IMU rows={len(imu)}, GT rows={len(data)})"
        )

    valid = np.isfinite(imu.to_numpy(dtype=float)).all(axis=1)
    valid &= np.isfinite(data.to_numpy(dtype=float)).all(axis=1)
    imu = imu.loc[valid].reset_index(drop=True)
    data = data.loc[valid].reset_index(drop=True)
    if imu.empty:
        raise ValueError("No finite, aligned rows remain after conversion")

    output_dir.mkdir(parents=True, exist_ok=True)
    imu.to_csv(output_dir / "imu.csv", index=False)
    data.to_csv(output_dir / "data.csv", index=False)
    return len(imu)


def create_plots(output_dir: Path) -> None:
    run_checked(
        [
            sys.executable,
            SCRIPT_DIR / "2_plot_csv_data.py",
            "--imu_csv",
            output_dir / "imu.csv",
            "--gt_csv",
            output_dir / "data.csv",
            "--out_dir",
            output_dir / "plots",
        ]
    )


def convert_job(job: BagJob, args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="rosbag_csv_") as temp_name:
        temp = Path(temp_name)
        raw_imu = temp / "raw_imu.csv"
        raw_gt = temp / "raw_gt.csv"
        clean_imu = temp / "raw_imu_clean.csv"
        clean_gt = temp / "raw_gt_clean.csv"
        aligned_imu = temp / "imu_aligned.csv"
        aligned_gt = temp / "gt_aligned.csv"

        run_checked(
            [
                sys.executable,
                SCRIPT_DIR / "1_rosbag_to_csv_vrpn.py",
                "--bag",
                job.bag,
                "--imu_out",
                raw_imu,
                "--gt_out",
                raw_gt,
            ]
        )

        clean_command: list[object] = [
            sys.executable,
            SCRIPT_DIR / "3_csv_dataset_cleaner.py",
            "--imu_csv",
            raw_imu,
            "--gt_csv",
            raw_gt,
            "--savgol_window",
            args.savgol_window,
            "--savgol_polyorder",
            args.savgol_polyorder,
        ]
        for bound in args.clean_bound:
            clean_command.extend(["--bound", bound])
        run_checked(clean_command)

        run_checked(
            [
                sys.executable,
                SCRIPT_DIR / "4_align_gt_and_imu.py",
                "--imu_csv",
                clean_imu,
                "--gt_csv",
                clean_gt,
                "--resample_imu_hz",
                args.rate,
                "--max_interp_gap",
                args.max_interp_gap,
                "--vicon_latency",
                args.vicon_latency,
                "--imu_resampled_out",
                aligned_imu,
                "--out",
                aligned_gt,
            ]
        )

        # Write only after every conversion stage succeeds.
        rows = write_dataset(aligned_imu, aligned_gt, job.output_dir)
        if not args.no_plots:
            create_plots(job.output_dir)
        return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.csv"
    columns = [
        "group",
        "flight_number",
        "source_bag",
        "output_dir",
        "rows",
        "status",
        "error",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest: {manifest}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert ROS 1 bags into cleaned, aligned CSV datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Rosbag directories; group names are inferred from directory names",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add an explicitly named input group; may be supplied multiple times",
    )
    parser.add_argument(
        "--all-known",
        action="store_true",
        help=(
            "Process the repository's low, medium, high, super-high, hover, "
            "and yaw folders"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "csv_datasets",
        help="Main output folder",
    )
    parser.add_argument("--rate", type=float, default=200.0, help="Output rate in Hz")
    parser.add_argument(
        "--max-interp-gap",
        type=float,
        default=0.05,
        help="Do not interpolate IMU across gaps larger than this many seconds",
    )
    parser.add_argument(
        "--vicon-latency",
        type=float,
        default=0.025,
        help="Seconds subtracted from Vicon timestamps before alignment",
    )
    parser.add_argument("--savgol-window", type=int, default=21)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument(
        "--clean-bound",
        action="append",
        default=[],
        metavar="PATTERN:MIN:MAX",
        help="Override a cleaner bound; may be supplied multiple times",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess flights whose imu.csv and data.csv already exist",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not create a plots subdirectory in each flight folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show input/output mappings without converting anything",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be greater than zero")
    if args.max_interp_gap <= 0:
        parser.error("--max-interp-gap must be greater than zero")

    groups: list[tuple[str, Path]] = []
    groups.extend((infer_group(path), path) for path in args.inputs)
    for spec in args.group:
        try:
            groups.append(parse_group_spec(spec))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    if args.all_known:
        groups.extend(KNOWN_GROUPS.items())
    if not groups:
        parser.error("provide one or more input paths, --group entries, or --all-known")

    output_root = args.output_root.expanduser().resolve()
    try:
        jobs = build_jobs(groups, output_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Found {len(jobs)} bag(s). Output root: {output_root}")
    for job in jobs:
        print(f"  [{job.group}] {job.bag} -> {job.output_dir}")
    if args.dry_run:
        return 0

    required_scripts = [
        SCRIPT_DIR / "1_rosbag_to_csv_vrpn.py",
        SCRIPT_DIR / "3_csv_dataset_cleaner.py",
        SCRIPT_DIR / "4_align_gt_and_imu.py",
    ]
    if not args.no_plots:
        required_scripts.append(SCRIPT_DIR / "2_plot_csv_data.py")
    missing = [str(path) for path in required_scripts if not path.is_file()]
    if missing:
        parser.error(f"required script(s) missing: {', '.join(missing)}")

    manifest_rows: list[dict[str, object]] = []
    failures = 0
    for index, job in enumerate(jobs, start=1):
        imu_out = job.output_dir / "imu.csv"
        data_out = job.output_dir / "data.csv"
        row: dict[str, object] = {
            "group": job.group,
            "flight_number": job.flight_number,
            "source_bag": str(job.bag),
            "output_dir": str(job.output_dir),
            "rows": "",
            "status": "",
            "error": "",
        }

        print(f"\n[{index}/{len(jobs)}] {job.bag}")
        if not args.overwrite and imu_out.is_file() and data_out.is_file():
            try:
                row["rows"] = len(pd.read_csv(imu_out, usecols=["time"]))
                if args.no_plots:
                    row["status"] = "skipped_existing"
                    print(f"    Skipping existing dataset: {job.output_dir}")
                else:
                    create_plots(job.output_dir)
                    row["status"] = "plotted_existing"
                    print(f"    Plotted existing dataset: {job.output_dir}")
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                failures += 1
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"    ERROR: {exc}", file=sys.stderr)
        else:
            try:
                row["rows"] = convert_job(job, args)
                row["status"] = "ok"
                print(f"    Wrote {row['rows']} rows to {job.output_dir}")
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                failures += 1
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"    ERROR: {exc}", file=sys.stderr)
        manifest_rows.append(row)

    write_manifest(output_root, manifest_rows)
    succeeded = sum(row["status"] == "ok" for row in manifest_rows)
    plotted = sum(row["status"] == "plotted_existing" for row in manifest_rows)
    skipped = sum(row["status"] == "skipped_existing" for row in manifest_rows)
    print(
        f"Completed: {succeeded} converted, {plotted} existing plotted, "
        f"{skipped} skipped, {failures} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
