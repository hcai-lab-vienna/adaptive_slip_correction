from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D
import json
import numpy as np
import os
import pandas as pd
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

DATASET_DIR = "/home/jbweibel/code/forest-mapping/adaptive_slip_correction/fomo-dataset_downloads/"
DEPLOYMENTS = [
    '2024-11-21',
    '2024-11-28',
    '2025-01-10',
    '2025-01-29',
    '2025-03-10',
    '2025-04-15',
    '2025-05-28',
    '2025-06-26',
    '2025-08-20',
    '2025-09-24',
    '2025-10-14',
    '2025-11-03',
]

TRAJECTORIES = [
    'red',
    'blue',
    'green',
    'magenta',
    'yellow',
    'orange'
]


def get_trajectory_dir(deployment="2024-11-21", trajectory="blue"):
    deployment_dir = os.path.join(DATASET_DIR, deployment)
    assert deployment in DEPLOYMENTS, f"Invalid deployment, Received '{deployment}'"
    assert trajectory in TRAJECTORIES, f"Invalid trajectory, Received '{trajectory}'"
    traj_dirs = [
        os.path.join(deployment_dir, fdir)
        for fdir in os.listdir(os.path.join(deployment_dir))
        if fdir.startswith(trajectory)
    ]
    assert len(traj_dirs) != 0, f"Invalid deployment-trajectory, Received '{deployment}' - '{trajectory}'. Valid options are {[d.split('_')[0] for d in os.listdir(os.path.join(deployment_dir))]}"

    return traj_dirs[0]


def get_transforms(trajectory_dir):
    transform_manager = TransformManager()

    with open(os.path.join(trajectory_dir, 'calib', 'transforms.json')) as fp:
        transforms = json.load(fp)

    for tf in transforms:
        t = np.array(
            [tf["position"]["x"], tf["position"]["y"], tf["position"]["z"]]
        )
        q = np.array(
            [
                tf["orientation"]["w"],
                tf["orientation"]["x"],
                tf["orientation"]["y"],
                tf["orientation"]["z"],
            ]
        )
        tf_mat = pt.transform_from_pq(np.hstack((t, q)))
        # we need this since pytransform3d assumes the tf is in the to_frame coordinate system,
        # while us (and ROS) expects the from (header) coordinate system
        tf_mat = np.linalg.inv(tf_mat)
        transform_manager.add_transform(tf["from"], tf["to"], tf_mat)

    return transform_manager


def get_gt_trajectory(trajectory_dir):
    return file_interface.read_tum_trajectory_file(os.path.join(trajectory_dir, 'gt.txt'))


def get_odom_trajectory(trajectory_dir):
    odom = pd.read_csv(os.path.join(trajectory_dir, 'odom.csv'))
    odom_traj = PoseTrajectory3D(
        odom[['px', 'py', 'pz']],
        odom[['qw', 'qx', 'qy', 'qz']],
        odom['t'] / 1e6)

    return (
        odom_traj,
        np.array(odom[['tlx', 'tly', 'tlz']]),
        np.array(odom[['tax', 'tay', 'taz']])
    )


def get_imu_data(trajectory_dir, imu='vectornav', tm=None):
    assert imu in ['vectornav', 'xsens']

    df = pd.read_csv(os.path.join(trajectory_dir, f'{imu}.csv'))
    accel = np.array(df.loc[:, ['ax', 'ay', 'az']])
    gyro = np.array(df.loc[:, ['wx', 'wy', 'wz']])

    with open(os.path.join(trajectory_dir, 'calib', 'imu.json')) as fp:
        calib = json.load(fp)[imu]['angular_velocity']
        gyro -= np.array([calib['x'], calib['y'], calib['z']])

    if tm:
        rot = tm.get_transform(imu, 'base_link')[:3,:3]
        accel = (rot @ accel.T).T
        gyro = (rot @ gyro.T).T

    return accel, gyro, np.array(df.loc[:, ['t']] / 1e6)


if __name__ == "__main__":
    trajectory_dir = get_trajectory_dir()
    transform_manager = get_transforms(trajectory_dir)
    get_gt_trajectory(trajectory_dir)
    get_odom_trajectory(trajectory_dir)
    get_imu_data(trajectory_dir)
