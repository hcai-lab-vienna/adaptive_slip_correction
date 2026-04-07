import numpy as np
import pandas as pd

from fomo_utils import (
    DEPLOYMENTS,
    TRAJECTORIES,
    get_trajectory_dir,
    get_transforms,
    get_odom_trajectory,
    get_robot_cmd,
    get_gt_trajectory,
    get_imu_data,
)
from imu_utils import mahony_filter, gravity_from_attitude, augment_odometry_with_imu
from trajectory_utils import (
    orientations_from_positions,
    sync,
    reduce_to_ids,
    relative_pose_from_trajectories,
    velocities_from_deltaT
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Adaptive slip correction.")
    parser.add_argument("-d", "--deployment", type=str, default="2024-11-21",
                        help=f"Deployment folder. Valid options are {DEPLOYMENTS}")
    parser.add_argument("-t", "--trajectory", type=str, default="blue",
                        help=f"Trajectory name. Valid options are {TRAJECTORIES}")
    args = parser.parse_args()


    ###########################################################################
    ### Load data
    print("Load data")
    trajectory_dir = get_trajectory_dir(deployment=args.deployment, trajectory=args.trajectory)

    # Transforms
    transform_manager = get_transforms(trajectory_dir)

    # IMUs
    imu_name = 'vectornav'  # 'vectornav'
    accel, gyro, imu_timestamps = get_imu_data(trajectory_dir, imu=imu_name, tm=transform_manager)

    # Trajectories
    traj_gt = get_gt_trajectory(trajectory_dir)
    traj_odom, lin_vel_twist, ang_vel_twist = get_odom_trajectory(trajectory_dir)
    cmd_df = get_robot_cmd(trajectory_dir, return_df=True)

    ###########################################################################
    ### Compute properties
    # IMUs
    print("Compute IMU orientations")
    g_earth = 9.80665
    imu_quats = mahony_filter(accel, gyro, imu_timestamps, g=g_earth, kp=1.0, ki=0.3)
    g_body = gravity_from_attitude(imu_quats)
    accel_grav_compensated = accel + g_earth * g_body

    # GT trajectory
    print("Process GT trajectory")
    delta_ts_gt = traj_gt.timestamps[1:] - traj_gt.timestamps[:-1]
    p_rel_gt = relative_pose_from_trajectories([traj_gt])[0]
    vel_gt = np.linalg.norm(np.array(p_rel_gt)[:,:3,3], axis=1) / delta_ts_gt
    gt_timestamps = traj_gt.timestamps[1:]

    # Wheel + IMU Odometry
    print("Process Odometry")
    ids_odom_to_imu, ids_imu_to_odom = sync.matching_time_indices(
        traj_odom.timestamps, imu_timestamps, max_diff=0.01)
    traj_odom_imu = augment_odometry_with_imu(
        reduce_to_ids(traj_odom, ids_odom_to_imu),
        reduce_to_ids(lin_vel_twist, ids_odom_to_imu),
        reduce_to_ids(imu_quats, ids_imu_to_odom)
    )
    p_rel_odom_imu = relative_pose_from_trajectories([traj_odom_imu])[0]
    delta_ts_odom_imu = traj_odom_imu.timestamps[1:] - traj_odom_imu.timestamps[:-1]
    vel_odom_imu = [velocities_from_deltaT(dT, dt) for dT, dt in zip(p_rel_odom_imu, delta_ts_odom_imu)]
    lin_vel_odom_imu = [vl for vl, va in vel_odom_imu]
    ang_vel_odom_imu = [va for vl, va in vel_odom_imu]

    ###########################################################################
    ### Align start-end and interpolate
    # Odom and Command
    odom_dt_index = pd.to_datetime(traj_odom_imu.timestamps[1:] * 1e9)
    cmd_df = cmd_df.reindex(odom_dt_index, method='nearest', tolerance=pd.Timedelta('50ms'))
    cmd_df = cmd_df.interpolate(method="linear", limit_direction="both")
    lin_vel_cmd = np.array(cmd_df[['lx', 'ly', 'lz']])
    ang_vel_cmd = np.array(cmd_df[['ax', 'ay', 'az']])

    # IMUs data
    accel_grav_compensated_sync = reduce_to_ids(accel_grav_compensated, ids_imu_to_odom)[2:]
    gyro_sync = reduce_to_ids(gyro, ids_imu_to_odom)[2:]
    g_body_sync = reduce_to_ids(g_body, ids_imu_to_odom)[2:]

    # GT data
    gt_df = pd.DataFrame({'ts': pd.to_datetime(1e9 * gt_timestamps), 'vel_gt': vel_gt})
    gt_df = gt_df.set_index('ts')
    gt_df = gt_df.reindex(odom_dt_index, method='nearest', tolerance=pd.Timedelta('50ms')).interpolate(method='time')

    features = pd.DataFrame(
        np.hstack([
            accel_grav_compensated_sync,
            gyro_sync,
            g_body_sync,
            lin_vel_odom_imu,
            ang_vel_odom_imu,
            lin_vel_cmd,
            ang_vel_cmd,
        ]),
        columns= [
            'lin_acc_imu_x',
            'lin_acc_imu_y',
            'lin_acc_imu_z',
            'ang_vel_imu_x',
            'ang_vel_imu_y',
            'ang_vel_imu_z',
            'grav_x',
            'grav_y',
            'grav_z',
            'lin_vel_odom_x',
            'lin_vel_odom_y',
            'lin_vel_odom_z',
            'ang_vel_odom_x',
            'ang_vel_odom_y',
            'ang_vel_odom_z',
            'lin_vel_cmd_x',
            'lin_vel_cmd_y',
            'lin_vel_cmd_z',
            'ang_vel_cmd_x',
            'ang_vel_cmd_y',
            'ang_vel_cmd_z',
        ]
    )

    print("Display trajectories")
    import matplotlib.pyplot as plt
    plt.plot(traj_odom_imu.positions_xyz[:,0], traj_odom_imu.positions_xyz[:,1])
    plt.plot(traj_odom.positions_xyz[:,0], traj_odom.positions_xyz[:,1])
    plt.plot(traj_gt.positions_xyz[:,0] - traj_gt.positions_xyz[0,0], traj_gt.positions_xyz[:,1] - traj_gt.positions_xyz[0, 1])

    plt.show()
