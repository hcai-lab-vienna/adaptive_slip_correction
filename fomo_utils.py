from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D
import json
import numpy as np
import os
import pandas as pd
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

DATASET_DIR = "/home/jbweibel/code/forest-mapping/adaptive_slip_correction/fomo-dataset_downloads/"

def get_trajectory_dir(deployment="2024-11-21", trajectory="blue"):
    deployment_dir = os.path.join(DATASET_DIR, deployment)
    return [
        os.path.join(deployment_dir, fdir)
        for fdir in os.listdir(os.path.join(deployment_dir))
        if fdir.startswith(trajectory)
    ][0]


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

    return odom_traj, odom[['t', 'tlx', 'tly', 'tlz', 'tax', 'tay', 'taz']]


if __name__ == "__main__":
    trajectory_dir = get_trajectory_dir()

    transform_manager = get_transforms(trajectory_dir)

    ### Obtain correctly oriented gravity vector
    # Load IMUs data
    df = pd.read_csv(os.path.join(trajectory_dir, 'vectornav.csv'))
    accel = np.array(df.loc[:, ['ax', 'ay', 'az']])
    gyro = np.array(df.loc[:, ['wx', 'wy', 'wz']])

    # Get the transformation between the vectornav and the robosense frame of reference
    # tf_vectornav_robosense = transform_manager.get_transform('vectornav', 'robosense')

    # Estimate the gravity vector
    freq = 200.0
    dt = 1/freq
    # g_body = estimate_gravity(accel, gyro, dt, g=9.80665, kp=2.0, ki=0.05)

    ### Correctly orient the path
    # Load odom and GT path
