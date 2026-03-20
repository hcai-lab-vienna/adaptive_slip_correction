from evo.core import metrics, sync, lie_algebra as lie
from evo.core.trajectory import PoseTrajectory3D
import numpy as np


def J_left_so3(phi):
    theta = np.linalg.norm(phi)
    I = np.eye(3)
    K = lie.hat(phi)
    if theta < 1e-8:
        # series approx
        return I + 0.5*K + (1/6.0)*(K @ K)
    A = (1 - np.cos(theta)) / (theta**2)
    B = (theta - np.sin(theta)) / (theta**3)
    return I + A*K + B*(K @ K)


def velocities_from_deltaT(DeltaT, dt):
    Rm = DeltaT[:3,:3]
    p = DeltaT[:3,3]
    phi = lie.so3_log(Rm)
    omega_b = phi / dt
    J = J_left_so3(phi)
    v_b = np.linalg.solve(J, p/dt)  # J^{-1} @ (p/dt)
    return v_b, omega_b  # body-frame linear and angular velocities


def deltaT_from_velocities(v_b, omega_b, dt):
    if type(v_b) != np.ndarray:
        v_b = np.array([v_b, 0.0, 0.0])

    if type(omega_b) != np.ndarray:
        omega_b = np.array([0.0, 0.0, omega_b])

    phi = omega_b * dt
    R = lie.so3_exp(phi)
    V = J_left_so3(phi)
    p = V @ (v_b * dt)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = p
    return T


def integrate_body_twists(v_b_list, omega_b_list, dt_list, T0=np.eye(4)):
    T_rels = []
    T_ws = [T0.copy()]
    for v_b, omega_b, dt in zip(v_b_list, omega_b_list, dt_list):
        T_rels.append(deltaT_from_velocities(v_b, omega_b, dt))
        T_ws.append(T_ws[-1] @ T_rels[-1])
    return T_rels, T_ws


def orientations_from_positions(traj, speed_eps=1e-3):
    delta_p = traj.positions_xyz[1:] - traj.positions_xyz[:-1]
    # np.vstack((
    #     (traj.positions_xyz[1] - traj.positions_xyz[0]).reshape(1,3),
    #     traj.positions_xyz[2:] - traj.positions_xyz[:-2],
    #     (traj.positions_xyz[-1] - traj.positions_xyz[-2]).reshape(1,3),
    # ))
    delta_ts = traj.timestamps[1:] - traj.timestamps[:-1]
    # np.hstack((
    #     (traj.timestamps[1:2] - traj.timestamps[0:1]),
    #     traj.timestamps[2:] - traj.timestamps[:-2],
    #     (traj.timestamps[-1:] - traj.timestamps[-2:-1])
    # ))
    # delta_p /= delta_ts.reshape((-1,1))

    headings = np.arctan2(delta_p[:,1], delta_p[:,0])
    low_speed_mask = np.linalg.norm(delta_p[1:], axis=1) < speed_eps
    headings[1:][low_speed_mask] = headings[:-1][low_speed_mask]
    headings[0] = 0.0 if np.linalg.norm(delta_p[0]) < speed_eps else headings[0]

    for idx, theta in enumerate(headings):
        traj.poses_se3[idx][:2, :2] = np.array(
            [[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]]
        )

    return traj


def relative_pose(traj, id_pairs):
    return [
        lie.relative_se3(traj.poses_se3[i], traj.poses_se3[j])
        for i, j in id_pairs
    ]


def relative_pose_from_sync_trajectories(
    trajectories,
    delta: float = 1,
    delta_unit: metrics.Unit = metrics.Unit.frames,
    rel_tol: float = 0.1):
    """
    Compute relative pose within a trajectory from a list of trajectories
    :param trajectories: tuple of evo trajectories assumed to by synchronized
    :param delta: the interval step for indices
    :param delta_unit: unit of delta (metrics.Unit enum member)
    :param rel_tol: relative tolerance to accept or reject deltas
    :return: list of index tuples (pairs)
    """
    id_pairs = metrics.id_pairs_from_delta(trajectories[0].poses_se3, delta, delta_unit, rel_tol)
    return [
        relative_pose(traj, id_pairs)
        for traj in trajectories
    ]


def compute_rpe_from_rel_pose(p_rel_ref, p_rel_est, pose_relation='translation'):
    E = [
        lie.relative_se3(p_rel_ref[i], p_rel_est[i])
        for i in range(len(p_rel_ref))
    ]

    pose_relation = {
        'translation': metrics.PoseRelation.translation_part,
        'rotation': metrics.PoseRelation.rotation_part,
        'full': metrics.PoseRelation.full_transformation,
        'rotation_angle_rad': metrics.PoseRelation.rotation_angle_rad,
        'rotation_angle_deg': metrics.PoseRelation.rotation_angle_deg,
    }[pose_relation]

    if pose_relation == metrics.PoseRelation.translation_part:
        return np.array(
            [np.linalg.norm(E_i[:3, 3]) for E_i in E]
        )
    elif pose_relation == metrics.PoseRelation.rotation_part:
        # ideal: rot(E_i) = 3x3 identity
        return np.array(
            [
                np.linalg.norm(lie.so3_from_se3(E_i) - np.eye(3))
                for E_i in E
            ]
        )
    elif pose_relation == metrics.PoseRelation.full_transformation:
        # ideal: E_i = 4x4 identity
        return np.array(
            [np.linalg.norm(E_i - np.eye(4)) for E_i in E]
        )
    elif pose_relation == metrics.PoseRelation.rotation_angle_rad:
        return np.array(
            [abs(lie.so3_log_angle(E_i[:3, :3])) for E_i in E]
        )
    elif pose_relation == metrics.PoseRelation.rotation_angle_deg:
        return np.array(
            [abs(lie.so3_log_angle(E_i[:3, :3], True)) for E_i in self.E]
        )


if __name__ == "__main__":
    from fomo_utils import get_trajectory_dir, get_odom_trajectory, get_gt_trajectory

    # Load data
    trajectory_dir = get_trajectory_dir()
    traj_gt = get_gt_trajectory(trajectory_dir)
    traj_odom, twist_odom = get_odom_trajectory(trajectory_dir)

    # Synchronize trajectories
    traj_gt_sync, traj_odom_sync = sync.associate_trajectories(traj_gt, traj_odom, 0.05)
    id_pairs = metrics.id_pairs_from_delta(traj_gt_sync.poses_se3, 1, metrics.Unit.frames, 0.1)

    traj_gt_oriented_sync = orientations_from_positions(traj_gt_sync)
    p_rel_gt, p_rel_odom = relative_pose_from_sync_trajectories([traj_gt_oriented_sync, traj_odom_sync])

    p_gt = np.array(traj_gt_sync.poses_se3)
    p_gt[:, :3, 3] -= traj_gt_sync.poses_se3[0][:3, 3]

    # Recover linear and angular velocities from trajectories
    delta_ts = traj_gt_sync.timestamps[1:] - traj_gt_sync.timestamps[:-1]
    vel_gt = [velocities_from_deltaT(dT, dt) for dT, dt in zip(p_rel_gt, delta_ts)]
    lin_vel_gt = [vl for vl, va in vel_gt]
    ang_vel_gt = [va for vl, va in vel_gt]

    vel_odom = [velocities_from_deltaT(dT, dt) for dT, dt in zip(p_rel_odom, delta_ts)]
    lin_vel_odom = [vl for vl, va in vel_odom]
    ang_vel_odom = [va for vl, va in vel_odom]

    lin_vel_err = [vl_gt - vl_odom for (vl_gt, _), (vl_odom, _)  in zip(vel_gt, vel_odom)]
    ang_vel_err = [va_gt - va_odom for (_, va_gt), (_, va_odom)  in zip(vel_gt, vel_odom)]

    import matplotlib.pyplot as plt
    plt.plot(lin_vel_err); plt.show()
    plt.plot(ang_vel_err); plt.show()

    # Reconstruct the trajectory from the velocities extracted
    p_rel_gt_rec, p_gt_rec = integrate_body_twists(lin_vel_gt, ang_vel_gt, delta_ts)
    p_rel_odom_rec, p_odom_rec = integrate_body_twists(lin_vel_odom, ang_vel_odom, delta_ts)

    print("GT rec err", compute_rpe_from_rel_pose(p_rel_gt, p_rel_gt_rec).mean())
    print("Odom rec err", compute_rpe_from_rel_pose(p_rel_odom, p_rel_odom_rec).mean())
