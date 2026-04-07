import copy
from evo.core import metrics, sync, lie_algebra as lie
from evo.core.trajectory import PoseTrajectory3D
import numpy as np


def kabsch_algorithm(
    traj1, traj2, alignment_frac: float = 0.25
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Modified Kabsch algorithm that fixes first points and finds optimal rotation.

    Parameters:
    traj1, traj2: numpy arrays of shape (n_points, 3) representing 3D trajectories
    alignment_frac: fraction of points to use for alignment

    Returns:
    r_a: rotation matrix (3x3) for evo transform
    t_a: translation vector (3,) for evo transform
    """
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)

    # Translation to align first points
    t_a = traj1[0] - traj2[0]
    traj1_centered = traj1 - traj1[0]
    traj2_centered = traj2 - traj2[0]

    target_len = int(alignment_frac * traj1_centered.shape[0])
    P = traj1_centered[0:target_len].T  # 3 x (n-1) - target points
    Q = traj2_centered[0:target_len].T  # 3 x (n-1) - points to rotate

    # Compute cross-covariance matrix, its SVD, and extract rotatio
    H = Q @ P.T
    U, S, Vt = np.linalg.svd(H)
    r_a = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(r_a) < 0:
        Vt[-1, :] *= -1
        r_a = Vt.T @ U.T

    return target_len, r_a, t_a


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


def velocities_from_trajectories(traj):
    traj = copy.deepcopy(traj)
    p_rel = relative_pose_from_trajectories([traj])[0]
    delta_ts = traj.timestamps[1:] - traj.timestamps[:-1]
    assert len(p_rel) == len(delta_ts), f"Length does not match, obtained {len(p_rel)}, {len(delta_ts)}"
    vel_gt = [velocities_from_deltaT(dT, dt) for dT, dt in zip(p_rel, delta_ts)]
    traj.reduce_to_ids(np.arange(1,len(traj.positions_xyz)))
    return (
        [vl for vl, va in vel_gt],
        [va for vl, va in vel_gt],
        traj,
        p_rel,
        delta_ts
    )


def integrate_body_twists(v_b_list, omega_b_list, dt_list, T0=np.eye(4)):
    T_rels = []
    T_ws = [T0.copy()]
    assert len(v_b_list) == len(omega_b_list) and len(v_b_list) == len(dt_list)
    for v_b, omega_b, dt in zip(v_b_list, omega_b_list, dt_list):
        T_rels.append(deltaT_from_velocities(v_b, omega_b, dt))
        T_ws.append(T_ws[-1] @ T_rels[-1])
    return T_rels, T_ws


def orientations_from_positions(traj, speed_eps=2e-1):
    traj = copy.deepcopy(traj)
    delta_p = traj.positions_xyz[1:] - traj.positions_xyz[:-1]
    delta_t = traj.timestamps[1:] - traj.timestamps[:-1]

    # # Smooth position delta, low velocity leads to delta close to noise in position
    smoothing_win_sz = 5
    delta_p[:, 0] = np.convolve(delta_p[:,0], np.ones(smoothing_win_sz) / smoothing_win_sz, mode="same")
    delta_p[:, 1] = np.convolve(delta_p[:, 1], np.ones(smoothing_win_sz) / smoothing_win_sz, mode="same")

    headings = np.arctan2(delta_p[:,1], delta_p[:,0])
    speed = np.linalg.norm(delta_p / delta_t.reshape((-1,1)), axis=1)
    for pos_idx in range(1, len(delta_p)):
        if speed[pos_idx] < speed_eps:
            headings[pos_idx] = headings[pos_idx-1]

    # Remove jumps in headings, as they will lead to spikes in angular velocity
    win_sz = 3
    headings = list(headings)[:win_sz] + [headings[i-win_sz:i+win_sz][win_sz] for i in range(win_sz, len(headings) - win_sz+1)] + list(headings)[-win_sz+1:]
    headings = [headings[0]] + headings
    for idx, theta in enumerate(headings):
        traj.poses_se3[idx][:2, :2] = np.array(
            [[np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]]
        )

    return traj, headings


def relative_pose(traj, id_pairs):
    return [
        lie.relative_se3(traj.poses_se3[i], traj.poses_se3[j])
        for i, j in id_pairs
    ]


def relative_pose_from_trajectories(
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


def reduce_to_ids(arr, ids):
    if type(arr) == PoseTrajectory3D:
        traj = copy.deepcopy(arr)
        traj.reduce_to_ids(ids)
        return traj
    elif type(arr) == np.ndarray:
        return arr[ids]
    elif type(arr) == list:
        return np.array(arr)[ids]
    else:
        raise Exception("Unknown type, reduce_to_ids only handles PoseTrajectory3D, list or np.ndarray")


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


def rmse(arr1, arr2):
    return np.linalg.norm(
        np.array(arr1) - np.array(arr2),
        axis=1
    )


if __name__ == "__main__":
    from fomo_utils import get_trajectory_dir, get_odom_trajectory, get_gt_trajectory

    # Load data
    trajectory_dir = get_trajectory_dir("2024-11-21", 'blue')
    traj_gt = get_gt_trajectory(trajectory_dir)
    # traj_gt_oriented, gt_headings = orientations_from_positions(traj_gt)
    traj_gt_oriented = copy.deepcopy(traj_gt)
    traj_odom, lin_vel_twist, ang_vel_twist = get_odom_trajectory(trajectory_dir)
    delta_ts_twist = traj_odom.timestamps[1:] - traj_odom.timestamps[:-1]

    # Recover linear and angular velocities from trajectories
    lin_vel_gt, ang_vel_gt, traj_gt_oriented, p_rel_gt, delta_ts = velocities_from_trajectories(traj_gt_oriented)
    vel = np.linalg.norm(np.array(p_rel_gt)[:,:3,3], axis=1) / delta_ts

    # Synchronize trajectories
    ids_gt, ids_odom = sync.matching_time_indices(traj_gt_oriented.timestamps, traj_odom.timestamps, max_diff=0.05)
    lin_vel_gt, ang_vel_gt = reduce_to_ids(lin_vel_gt, ids_gt), reduce_to_ids(ang_vel_gt, ids_gt)
    p_rel_gt, delta_ts = reduce_to_ids(p_rel_gt, ids_gt), reduce_to_ids(delta_ts, ids_gt)
    traj_gt_sync = reduce_to_ids(traj_gt_oriented, ids_gt)
    traj_odom_sync = reduce_to_ids(traj_odom, ids_odom)
    lin_vel_twist_sync = lin_vel_twist[ids_odom]
    ang_vel_twist_sync = ang_vel_twist[ids_odom]

    print("Twist-GT linear velocity RMSE", rmse(lin_vel_gt, lin_vel_twist_sync).mean())
    print("Twist-GT angular velocity RMSE", rmse(ang_vel_gt, ang_vel_twist_sync).mean())

    ###########################################################################
    # Test the velocity from poses computation
    ###########################################################################

    lin_vel_odom, ang_vel_odom, traj_odom, p_rel_odom, delta_ts_odom = velocities_from_trajectories(traj_odom)
    ids_gt, ids_odom = sync.matching_time_indices(traj_gt_oriented.timestamps, traj_odom.timestamps, max_diff=0.05)

    import matplotlib.pyplot as plt
    eps = 1e-1
    plt.plot([(vgt- v[0]) / max(vgt, eps) for (v, vgt) in zip(lin_vel_twist_sync, vel)]);
    plt.plot([v[0] for (v, vgt) in zip(lin_vel_twist_sync, vel)]);
    plt.plot([vgt - 2. for (v, vgt) in zip(lin_vel_twist_sync, vel)]);
    plt.show()

    # traj_odom lost the first element in the velocity computation, so we have to index from 1
    print("Odom-Twist linear velocity RMSE full traj.", rmse(lin_vel_twist[1:], lin_vel_odom).mean())
    print("Odom-Twist angular velocity RMSE full traj.", rmse(ang_vel_twist[1:], ang_vel_odom).mean())

    ###########################################################################
    # Test the poses from velocites computation
    ###########################################################################
    p_rel_gt_rec, p_gt_rec = integrate_body_twists(lin_vel_gt, ang_vel_gt, delta_ts)
    p_rel_odom_rec, p_odom_rec = integrate_body_twists(lin_vel_odom, ang_vel_odom, delta_ts_odom)
    p_rel_twist, p_twist_rec = integrate_body_twists(lin_vel_twist[1:], ang_vel_twist[1:], delta_ts_twist)


    print("GT  reconstruction RPE", compute_rpe_from_rel_pose(p_rel_gt, p_rel_gt_rec, 'full').mean())
    print("Odom reconstruction RPE", compute_rpe_from_rel_pose(p_rel_odom, p_rel_odom_rec, 'full').mean())
    print("Odom-Twist RPE", compute_rpe_from_rel_pose(p_rel_odom, p_rel_twist, 'full').mean())

    traj_gt_aligned = copy.deepcopy(traj_gt_sync)
    num_used_poses, r_a, t_a = kabsch_algorithm(
        np.array(p_gt_rec)[1:,:3,3], traj_gt_aligned.positions_xyz
    )
    pos_gt_aligned = np.dot(r_a, (traj_gt_aligned.positions_xyz + t_a).T).T
    print("GT absolute position reconstruction RMSE", rmse(np.array(p_gt_rec)[1:, :3, 3], pos_gt_aligned).mean())
    print("Odom absolute position reconstruction RMSE", rmse(np.array(p_odom_rec)[:, :3, 3], np.array(p_twist_rec)[:, :3, 3]).mean())
