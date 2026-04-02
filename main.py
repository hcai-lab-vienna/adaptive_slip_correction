from fomo_utils import (
    DEPLOYMENTS,
    TRAJECTORIES,
    get_trajectory_dir,
    get_transforms,
    get_odom_trajectory,
    get_gt_trajectory,
    get_imu_data,
)
from imu_utils import mahony_filter, gravity_from_attitude, augment_odometry_with_imu
from trajectory_utils import (orientations_from_positions,
                              sync,
                              reduce_to_ids,
                              velocities_from_trajectories)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Adaptive slip correction.")
    parser.add_argument("-d", "--deployment", type=str, default="2024-11-21",
                        help=f"Deployment folder. Valid options are {DEPLOYMENTS}")
    parser.add_argument("-t", "--trajectory", type=str, default="blue",
                        help=f"Trajectory name. Valid options are {TRAJECTORIES}")
    args = parser.parse_args()

    ### Load data
    trajectory_dir = get_trajectory_dir(deployment=args.deployment, trajectory=args.trajectory)
    # Transforms
    transform_manager = get_transforms(trajectory_dir)
    # IMUs
    imu_name = 'xsens'
    accel, gyro, imu_timestamps = get_imu_data(trajectory_dir, imu=imu_name, tm=transform_manager)
    # Trajectories
    traj_gt = get_gt_trajectory(trajectory_dir)
    traj_gt_oriented, gt_headings = orientations_from_positions(traj_gt)
    traj_odom, lin_vel_twist, ang_vel_twist = get_odom_trajectory(trajectory_dir)
    delta_ts_twist = traj_odom.timestamps[1:] - traj_odom.timestamps[:-1]

    # Recover linear and angular velocities from GT trajectory
    lin_vel_gt, ang_vel_gt, traj_gt_oriented, p_rel_gt, delta_ts = velocities_from_trajectories(traj_gt_oriented)

    # Synchronize trajectories
    ids_gt, ids_odom = sync.matching_time_indices(traj_gt_oriented.timestamps, traj_odom.timestamps, max_diff=0.05)
    lin_vel_gt, ang_vel_gt = reduce_to_ids(lin_vel_gt, ids_gt), reduce_to_ids(ang_vel_gt, ids_gt)
    p_rel_gt, delta_ts = reduce_to_ids(p_rel_gt, ids_gt), reduce_to_ids(delta_ts, ids_gt)
    traj_gt_sync = reduce_to_ids(traj_gt_oriented, ids_gt)
    traj_odom_sync = reduce_to_ids(traj_odom, ids_odom)
    lin_vel_twist_sync = lin_vel_twist[ids_odom]
    ang_vel_twist_sync = ang_vel_twist[ids_odom]

    # Process the IMU
    freq = 200.0
    imu_quats = mahony_filter(accel, gyro, imu_timestamps, g=9.80665, kp=1.0, ki=0.3)
    # from ahrs.filters import Mahony
    # orientation = Mahony(gyr=gyro, acc=accel, frequency=freq)

    # Estimate the gravity vector
    # g_body = gravity_from_attitude(imu_attitude_qs)

    ids_odom_to_imu, ids_imu_to_odom = sync.matching_time_indices(
        traj_odom.timestamps, imu_timestamps, max_diff=0.01)
    traj_odom_imu = augment_odometry_with_imu(
        reduce_to_ids(traj_odom, ids_odom_to_imu),
        reduce_to_ids(lin_vel_twist, ids_odom_to_imu),
        reduce_to_ids(imu_quats, ids_imu_to_odom)
    )

    import matplotlib.pyplot as plt
    plt.plot(traj_odom_imu.positions_xyz[:,0], traj_odom_imu.positions_xyz[:,1])
    plt.plot(traj_odom.positions_xyz[:,0], traj_odom.positions_xyz[:,1])
    plt.plot(traj_gt.positions_xyz[:,0] - traj_gt.positions_xyz[0,0], traj_gt.positions_xyz[:,1] - traj_gt.positions_xyz[0, 1])

    plt.show()
