#!/usr/bin/env python3

import argparse
import os
import rosbag
import pandas as pd


def pose_message_to_row(t, msg, prefix):
    return {
        "time": t,
        f"{prefix}_x": msg.pose.position.x,
        f"{prefix}_y": msg.pose.position.y,
        f"{prefix}_z": msg.pose.position.z,
        f"{prefix}_qx": msg.pose.orientation.x,
        f"{prefix}_qy": msg.pose.orientation.y,
        f"{prefix}_qz": msg.pose.orientation.z,
        f"{prefix}_qw": msg.pose.orientation.w,
    }


def imu_message_to_row(t, msg):
    return {
        "time": t,
        "acc_x": msg.linear_acceleration.x,
        "acc_y": msg.linear_acceleration.y,
        "acc_z": msg.linear_acceleration.z,
        "gyro_x": msg.angular_velocity.x,
        "gyro_y": msg.angular_velocity.y,
        "gyro_z": msg.angular_velocity.z,
        "qx": msg.orientation.x,
        "qy": msg.orientation.y,
        "qz": msg.orientation.z,
        "qw": msg.orientation.w,
    }


def twist_message_to_row(t, msg):
    return {
        "time": t,
        "vel_x": msg.twist.linear.x,
        "vel_y": msg.twist.linear.y,
        "vel_z": msg.twist.linear.z,
    }


def extract_columns(df, columns):
    if df.empty or any(col not in df.columns for col in columns):
        return pd.DataFrame(columns=columns)
    return df[columns]


def build_ground_truth_dataframe(df_pose, pose_prefix, df_vel, tolerance):
    columns = [
        "time",
        "pos_x",
        "pos_y",
        "pos_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "vel_x",
        "vel_y",
        "vel_z",
    ]

    if df_pose.empty:
        return pd.DataFrame(columns=columns)

    pose_cols = {
        f"{pose_prefix}_x": "pos_x",
        f"{pose_prefix}_y": "pos_y",
        f"{pose_prefix}_z": "pos_z",
        f"{pose_prefix}_qw": "quat_w",
        f"{pose_prefix}_qx": "quat_x",
        f"{pose_prefix}_qy": "quat_y",
        f"{pose_prefix}_qz": "quat_z",
    }
    base_cols = ["time", *pose_cols.keys()]
    if any(col not in df_pose.columns for col in base_cols):
        return pd.DataFrame(columns=columns)

    gt = df_pose[base_cols].rename(columns=pose_cols).sort_values("time")

    if not df_vel.empty:
        gt = pd.merge_asof(
            gt,
            df_vel.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=tolerance,
        )

    return gt.reindex(columns=columns)


def derive_output_paths(bag_path, imu_out, gt_out):
    bag_dir = os.path.dirname(bag_path)
    bag_base = os.path.splitext(os.path.basename(bag_path))[0]
    imu_default = os.path.join(bag_dir, f"{bag_base}_imu.csv")
    gt_default = os.path.join(bag_dir, f"{bag_base}_gt.csv")
    return imu_out or imu_default, gt_out or gt_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--imu_out")
    parser.add_argument("--gt_out")
    parser.add_argument(
        "--velocity_topic",
        default="/mavros/vision_speed/speed_twist",
        help="Uses twist/linear/{x,y,z}",
    )
    parser.add_argument(
        "--imu_topic",
        default="/mavros/imu/data_raw",
        help="Uses angular_velocity/{x,y,z} and linear_acceleration/{x,y,z}",
    )
    parser.add_argument(
        "--vicon_topic",
        default="/mavros/vision_pose/pose",
        help="Uses pose/position/{x,y,z} and pose/orientation/{w,x,y,z}",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    args.imu_out, args.gt_out = derive_output_paths(
        args.bag,
        args.imu_out,
        args.gt_out,
    )

    velocity_rows = []
    imu_rows = []
    vicon_rows = []

    selected_topics = [
        args.velocity_topic, # velocity data (for GT)
        args.imu_topic,   # IMU data
        args.vicon_topic, # vicon position GT
    ]

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=selected_topics):
            t = stamp.to_sec()

            if topic == args.velocity_topic:
                velocity_rows.append(twist_message_to_row(t, msg))

            elif topic == args.imu_topic:
                imu_rows.append(imu_message_to_row(t, msg))

            elif topic == args.vicon_topic:
                vicon_rows.append(pose_message_to_row(t, msg, "vicon"))

    df_vel = pd.DataFrame(velocity_rows).sort_values("time")
    df_imu = pd.DataFrame(imu_rows).sort_values("time")
    df_vicon = pd.DataFrame(vicon_rows).sort_values("time")

    df = df_imu.copy()

    for other_df in [df_vel, df_vicon]:
        if len(other_df) == 0:
            continue

        df = pd.merge_asof(
            df.sort_values("time"),
            other_df.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=args.tolerance,
        )

    imu_columns = [
        "time",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "acc_x",
        "acc_y",
        "acc_z",
    ]
    df_imu_out = extract_columns(df_imu, imu_columns).sort_values("time")
    df_imu_out.to_csv(args.imu_out, index=False)
    print(f"Saved IMU CSV to {args.imu_out}")
    print(f"IMU rows: {len(df_imu_out)}")

    df_gt = build_ground_truth_dataframe(df_vicon, "vicon", df_vel, args.tolerance)

    df_gt.to_csv(args.gt_out, index=False)
    print(f"Saved GT CSV to {args.gt_out}")
    print(f"GT rows: {len(df_gt)}")

# python3 rosbag_to_csv.py --bag flight_6_27.bag 

if __name__ == "__main__":
    main()