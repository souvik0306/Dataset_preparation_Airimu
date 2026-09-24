#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import rosbag


IMU_COLUMNS = [
    "time",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
]

GT_COLUMNS = [
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


def header_time_to_sec(msg):
    """Return the message header timestamp in seconds."""
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        message_type = getattr(msg, "_type", type(msg).__name__)
        raise ValueError(
            f"message type {message_type} has no header.stamp timestamp"
        )
    return msg.header.stamp.to_sec()


def imu_message_to_row(t, msg):
    return {
        "time": t,
        "gyro_x": msg.angular_velocity.x,
        "gyro_y": msg.angular_velocity.y,
        "gyro_z": msg.angular_velocity.z,
        "acc_x": msg.linear_acceleration.x,
        "acc_y": msg.linear_acceleration.y,
        "acc_z": msg.linear_acceleration.z,
    }


def vicon_pose_message_to_row(t, msg):
    return {
        "time": t,
        "pos_x": msg.pose.position.x,
        "pos_y": msg.pose.position.y,
        "pos_z": msg.pose.position.z,
        "quat_w": msg.pose.orientation.w,
        "quat_x": msg.pose.orientation.x,
        "quat_y": msg.pose.orientation.y,
        "quat_z": msg.pose.orientation.z,
    }


def vicon_twist_message_to_row(t, msg):
    return {
        "time": t,
        "vel_x": msg.twist.linear.x,
        "vel_y": msg.twist.linear.y,
        "vel_z": msg.twist.linear.z,
    }


def dataframe_from_rows(rows, columns):
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns).sort_values("time")


def build_ground_truth_dataframe(df_pose, df_twist, tolerance):
    if df_pose.empty:
        return pd.DataFrame(columns=GT_COLUMNS)

    gt = df_pose.sort_values("time")
    if not df_twist.empty:
        gt = pd.merge_asof(
            gt,
            df_twist.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=tolerance,
        )

    return gt.reindex(columns=GT_COLUMNS)


def derive_output_paths(bag_path, imu_out, gt_out):
    bag_dir = os.path.dirname(bag_path)
    bag_base = os.path.splitext(os.path.basename(bag_path))[0]
    imu_default = os.path.join(bag_dir, f"{bag_base}_imu.csv")
    gt_default = os.path.join(bag_dir, f"{bag_base}_gt.csv")
    return imu_out or imu_default, gt_out or gt_default


def main():
    parser = argparse.ArgumentParser(
        description="Extract IMU and AIIMU1 VRPN ground-truth CSV files from a ROS bag.",
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--imu_out")
    parser.add_argument("--gt_out")
    parser.add_argument(
        "--imu_topic",
        default="/mavros/imu/data_raw",
        help="IMU topic; uses angular_velocity/{x,y,z} and linear_acceleration/{x,y,z}",
    )
    parser.add_argument(
        "--vicon_pose_topic",
        default="/vrpn_client_node/AIIMU1/pose",
        help="VRPN pose topic; uses pose/position/{x,y,z} and pose/orientation/{w,x,y,z}",
    )
    parser.add_argument(
        "--vicon_twist_topic",
        default="/vrpn_client_node/AIIMU1/twist",
        help="VRPN twist topic; uses twist/linear/{x,y,z}",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Nearest-neighbor tolerance in seconds when matching VRPN pose and twist",
    )
    args = parser.parse_args()

    args.imu_out, args.gt_out = derive_output_paths(
        args.bag,
        args.imu_out,
        args.gt_out,
    )

    imu_rows = []
    vicon_pose_rows = []
    vicon_twist_rows = []

    selected_topics = [
        args.imu_topic,
        args.vicon_pose_topic,
        args.vicon_twist_topic,
    ]

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, _bag_stamp in bag.read_messages(topics=selected_topics):
            t = header_time_to_sec(msg)

            if topic == args.imu_topic:
                imu_rows.append(imu_message_to_row(t, msg))
            elif topic == args.vicon_pose_topic:
                vicon_pose_rows.append(vicon_pose_message_to_row(t, msg))
            elif topic == args.vicon_twist_topic:
                vicon_twist_rows.append(vicon_twist_message_to_row(t, msg))

    df_imu = dataframe_from_rows(imu_rows, IMU_COLUMNS)
    df_pose = dataframe_from_rows(
        vicon_pose_rows,
        ["time", "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"],
    )
    df_twist = dataframe_from_rows(
        vicon_twist_rows,
        ["time", "vel_x", "vel_y", "vel_z"],
    )
    df_gt = build_ground_truth_dataframe(df_pose, df_twist, args.tolerance)

    df_imu.to_csv(args.imu_out, index=False)
    print(f"Saved IMU CSV to {args.imu_out}")
    print(f"IMU rows: {len(df_imu)}")

    df_gt.to_csv(args.gt_out, index=False)
    print(f"Saved GT CSV to {args.gt_out}")
    print(f"GT rows: {len(df_gt)}")
    print(f"VRPN pose rows: {len(df_pose)}")
    print(f"VRPN twist rows: {len(df_twist)}")


if __name__ == "__main__":
    main()
