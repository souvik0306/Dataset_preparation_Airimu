#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Edit these arrays to control which columns are plotted.
IMU_PLOT_COLS = [
	"gyro_x",
	"gyro_y",
	"gyro_z",
	"acc_x",
	"acc_y",
	"acc_z",
]

GT_PLOT_COLS = [
	"vel_x",
	"vel_y",
	"vel_z",
    "quat_w",
	"quat_x",
	"quat_y",
	"quat_z"
]

GT_POSITION_COLS = [
	"pos_x",
	"pos_y",
	"pos_z",
]


def _load_csv(path: str, required_time_col: str = "time") -> pd.DataFrame:
	df = pd.read_csv(path)
	if required_time_col not in df.columns:
		raise ValueError(f"Missing required column '{required_time_col}' in {path}")
	return df.sort_values(required_time_col)


def _filter_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
	return [col for col in columns if col in df.columns]


def _plot_series(
	df: pd.DataFrame,
	columns: List[str],
	label_prefix: str,
	out_dir: Path,
	show: bool,
) -> None:
	available = _filter_columns(df, columns)
	if not available:
		print(f"No matching columns found for: {label_prefix}")
		return

	time_values = df["time"].to_numpy()
	if len(time_values) > 0:
		time_values = time_values - time_values[0]

	for col in available:
		plt.figure(figsize=(12, 6))
		plt.plot(time_values, df[col].to_numpy(), label=col)
		plt.title(f"{label_prefix}: {col}")
		plt.xlabel("time (from start)")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		output_path = out_dir / f"{label_prefix.lower()}_{col}.png"
		plt.savefig(output_path, dpi=150)
		if not show:
			plt.close()


def _plot_subplots(
	df: pd.DataFrame,
	columns: List[str],
	title: str,
	output_name: str,
	out_dir: Path,
	show: bool,
) -> None:
	available = _filter_columns(df, columns)
	if not available:
		print(f"No matching columns found for: {title}")
		return

	time_values = df["time"].to_numpy()
	if len(time_values) > 0:
		time_values = time_values - time_values[0]

	fig, axes = plt.subplots(
		len(available),
		1,
		figsize=(12, 3 * len(available)),
		sharex=True,
	)
	if len(available) == 1:
		axes = [axes]

	for axis, col in zip(axes, available):
		axis.plot(time_values, df[col].to_numpy(), label=col)
		axis.set_ylabel(col)
		axis.legend()
		axis.grid(True)

	axes[-1].set_xlabel("time (from start)")
	fig.suptitle(title)
	fig.tight_layout()
	output_path = out_dir / output_name
	fig.savefig(output_path, dpi=150)
	if not show:
		plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--imu_csv", default="imu.csv")
	parser.add_argument("--gt_csv", default="gt.csv")
	parser.add_argument("--show", action="store_true", help="Show plots interactively")
	parser.add_argument("--out_dir", default="plots_aligned_f627", help="Directory to save plots")
	args = parser.parse_args()

	imu_path = Path(args.imu_csv)
	gt_path = Path(args.gt_csv)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	if imu_path.exists():
		df_imu = _load_csv(str(imu_path))
		_plot_series(df_imu, IMU_PLOT_COLS, "IMU", out_dir, args.show)
	else:
		print(f"IMU CSV not found: {imu_path}")

	if gt_path.exists():
		df_gt = _load_csv(str(gt_path))
		_plot_subplots(
			df_gt,
			GT_POSITION_COLS,
			"GT position",
			"gt_position.png",
			out_dir,
			args.show,
		)
		_plot_series(df_gt, GT_PLOT_COLS, "GT", out_dir, args.show)
	else:
		print(f"GT CSV not found: {gt_path}")

	if args.show:
		plt.show()
	else:
		plt.close("all")


if __name__ == "__main__":
	main()

# python3 plot_csv_data.py --imu_csv flight_6_22_imu.csv --gt_csv flight_6_22_gt.csv --show
