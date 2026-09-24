#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_sorted_csv(path: Path, time_col: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	if time_col not in df.columns:
		raise ValueError(f"Missing required column '{time_col}' in {path}")

	df = df.copy()
	df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
	df = df.dropna(subset=[time_col]).sort_values(time_col)
	return df


def _apply_latency(df: pd.DataFrame, time_col: str, latency_s: float) -> pd.DataFrame:
	if latency_s == 0:
		return df
	shifted = df.copy()
	shifted[time_col] = shifted[time_col] - latency_s
	return shifted.sort_values(time_col)


def _derive_output_path(
	gt_path: Path,
	out_path: Optional[Path],
	out_dir: Optional[Path],
) -> Path:
	if out_path is not None:
		return out_path
	if out_dir is not None:
		return out_dir / f"{gt_path.stem}_aligned{gt_path.suffix}"
	return gt_path.with_name(f"{gt_path.stem}_aligned{gt_path.suffix}")


def _derive_imu_resampled_output_path(imu_path: Path, rate_hz: float) -> Path:
	rate_label = f"{rate_hz:g}".replace(".", "p")
	return imu_path.with_name(f"{imu_path.stem}_{rate_label}hz{imu_path.suffix}")


def _timeline_summary(df: pd.DataFrame, time_col: str) -> str:
	if df.empty:
		return "rows=0, duration=0.000s, rate=0.00 Hz"

	start = float(df[time_col].iloc[0])
	end = float(df[time_col].iloc[-1])
	duration = end - start
	rate = (len(df) - 1) / duration if duration > 0 and len(df) > 1 else 0.0
	return (
		f"rows={len(df)}, duration={duration:.3f}s, "
		f"rate={rate:.2f} Hz, start={start:.6f}, end={end:.6f}"
	)


def _interpolate_segment_to_rate(
	segment: pd.DataFrame,
	time_col: str,
	rate_hz: float,
	include_start: bool,
) -> pd.DataFrame:
	start = float(segment[time_col].iloc[0])
	end = float(segment[time_col].iloc[-1])
	dt = 1.0 / rate_hz
	first_time = start if include_start else start + dt

	if end < first_time:
		return segment.iloc[[0]].copy() if include_start else segment.iloc[0:0].copy()

	num_samples = int(np.floor((end - first_time) / dt)) + 1
	target_times = first_time + np.arange(num_samples) * dt
	output = pd.DataFrame({time_col: target_times})
	source_time = segment[time_col].to_numpy()

	for col in segment.columns:
		if col == time_col:
			continue
		if pd.api.types.is_numeric_dtype(segment[col]):
			output[col] = np.interp(target_times, source_time, segment[col].to_numpy())

	return output.reindex(columns=segment.columns)


def _resample_imu_to_rate(
	df: pd.DataFrame,
	time_col: str,
	rate_hz: Optional[float],
	max_gap_s: float,
) -> pd.DataFrame:
	if rate_hz is None or rate_hz <= 0 or df.empty:
		return df

	if len(df) == 1:
		return df.copy()

	df = df.sort_values(time_col).drop_duplicates(subset=[time_col]).reset_index(drop=True)
	gaps = df[time_col].diff().fillna(0)
	segment_ids = (gaps > max_gap_s).cumsum()

	segments = []
	for _, segment in df.groupby(segment_ids, sort=False):
		segments.append(
			_interpolate_segment_to_rate(
				segment,
				time_col,
				rate_hz,
				include_start=True,
			)
		)

	if not segments:
		return df

	return pd.concat(segments, ignore_index=True).sort_values(time_col)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compensate Vicon latency and align GT to the IMU timeline.",
	)
	parser.add_argument("--imu_csv", required=True, help="50 Hz EKF/IMU CSV")
	parser.add_argument("--gt_csv", required=True, help="100 Hz Vicon GT CSV")
	parser.add_argument("--out", help="Aligned GT output CSV path")
	parser.add_argument("--imu_resampled_out", help="Resampled IMU output CSV path")
	parser.add_argument("--out_dir", help="Output directory for aligned GT CSV")
	parser.add_argument("--time_col", default="time", help="Timestamp column name")
	parser.add_argument(
		"--resample_imu_hz",
		type=float,
		default=None,
		help="Resample/interpolate IMU to this fixed rate before aligning GT",
	)
	parser.add_argument(
		"--max_interp_gap",
		type=float,
		default=0.05,
		help="Reset IMU interpolation across gaps larger than this many seconds",
	)
	parser.add_argument(
		"--vicon_latency",
		type=float,
		default=0.025,
		help="Seconds to subtract from GT time to compensate transport lag",
	)
	parser.add_argument(
		"--tolerance",
		type=float,
		default=None,
		help="Optional max time delta (seconds) for merge_asof",
	)
	args = parser.parse_args()

	imu_path = Path(args.imu_csv)
	gt_path = Path(args.gt_csv)
	if not imu_path.exists():
		raise SystemExit(f"IMU CSV not found: {imu_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	out_dir = Path(args.out_dir) if args.out_dir else None
	if out_dir is not None:
		out_dir.mkdir(parents=True, exist_ok=True)

	output_path = _derive_output_path(
		gt_path,
		Path(args.out) if args.out else None,
		out_dir,
	)

	df_imu = _load_sorted_csv(imu_path, args.time_col)
	df_gt_raw = _load_sorted_csv(gt_path, args.time_col)
	df_imu_resampled = _resample_imu_to_rate(
		df_imu,
		args.time_col,
		args.resample_imu_hz,
		args.max_interp_gap,
	)

	if args.resample_imu_hz is not None and args.resample_imu_hz > 0:
		imu_resampled_path = (
			Path(args.imu_resampled_out)
			if args.imu_resampled_out
			else _derive_imu_resampled_output_path(imu_path, args.resample_imu_hz)
		)
		df_imu_resampled.to_csv(imu_resampled_path, index=False)

	# Step 3: deterministic latency compensation on the Vicon timeline.
	df_gt = _apply_latency(df_gt_raw, args.time_col, args.vicon_latency)

	# Step 4: align to the IMU timeline using nearest-neighbor matching.
	merge_kwargs = {
		"on": args.time_col,
		"direction": "nearest",
	}
	if args.tolerance is not None:
		merge_kwargs["tolerance"] = args.tolerance

	df_gt_for_merge = df_gt.copy()
	df_gt_for_merge["__gt_match"] = 1
	df_sync = pd.merge_asof(df_imu_resampled, df_gt_for_merge, **merge_kwargs)
	gt_aligned = df_sync[df_gt.columns]
	gt_aligned.to_csv(output_path, index=False)

	matched_rows = int(df_sync["__gt_match"].notna().sum())
	unmatched_rows = len(df_sync) - matched_rows

	print(f"Saved aligned GT CSV to {output_path}")
	if args.resample_imu_hz is not None and args.resample_imu_hz > 0:
		print(f"Saved resampled IMU CSV to {imu_resampled_path}")
	print("\nOriginal timelines:")
	print(f"  IMU: {_timeline_summary(df_imu, args.time_col)}")
	print(f"  GT:  {_timeline_summary(df_gt_raw, args.time_col)}")
	if args.resample_imu_hz is not None and args.resample_imu_hz > 0:
		print(f"  IMU interpolated to {args.resample_imu_hz:g} Hz:")
		print(f"       {_timeline_summary(df_imu_resampled, args.time_col)}")
	print(f"  GT after latency compensation ({args.vicon_latency:.6f}s):")
	print(f"       {_timeline_summary(df_gt, args.time_col)}")
	print("\nAligned output:")
	print(f"  GT aligned to IMU: {_timeline_summary(gt_aligned, args.time_col)}")
	print(f"  Matched IMU rows: {matched_rows}")
	print(f"  Unmatched IMU rows: {unmatched_rows}")


if __name__ == "__main__":
	main()
