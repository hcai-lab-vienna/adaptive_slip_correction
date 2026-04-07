import copy
from evo.core import transformations as tr
from evo.core.trajectory import PoseTrajectory3D
import numpy as np


def quat_exp_omega(q, omega, dt):
    # Integrate q_dot = 0.5*q⊗[0,omega], first-order
    omega_quat = np.hstack(([0.0], omega))
    dq = tr.quaternion_multiply(q, omega_quat) * 0.5 * dt
    q_new = q + dq
    return q_new / np.linalg.norm(q_new)


def init_quat_from_acc(a):
    # Quaternion from given acceleration.
    # Comes from:
    # https://github.com/Mayitzin/ahrs/blob/master/ahrs/common/orientation.py#L770
    q = np.array([1.0, 0.0, 0.0, 0.0])
    ex, ey, ez = 0.0, 0.0, 0.0
    if np.linalg.norm(a) > 0 and len(a) == 3:
        ax, ay, az = a
        # Normalize accelerometer measurements
        a_norm = np.linalg.norm(a)
        ax /= a_norm
        ay /= a_norm
        az /= a_norm
        # Euler Angles from Gravity vector
        ex = np.arctan2(ay, az)
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        ez = 0.0
        # Euler to Quaternion
        q = tr.quaternion_from_euler(ex, ey, ez)
        q /= np.linalg.norm(q)
    return q


def mahony_filter(acc, gyro, ts, g=9.80665, kp=1.0, ki=0.3, tol=0.3):
    # Inspired by Mahony filter from:
    # https://github.com/Mayitzin/ahrs/blob/master/ahrs/filters/mahony.py
    # Adding filter on accelerometer norm as the IMU is on a moving robot.
    N = len(acc)
    q = init_quat_from_acc(acc[0])
    bg = np.zeros(3)
    qs = np.zeros((N,4))

    for k in range(1, N):
        if np.linalg.norm(gyro[k]) == 0:
            qs[k] = q
            continue

        dt = ts[k] - ts[k-1]
        # Use accel if credible
        a = acc[k]
        anorm = np.linalg.norm(a)
        omega = gyro[k]

        # if 0.8*g < anorm < 1.2*g and np.linalg.norm(gyro[k]) < 3.0:
        if (1.-tol)*g < anorm < (1.+tol)*g:
            R = tr.quaternion_matrix(q)[:3,:3]  # world_from_body
            g_up_hat = R.T @ np.array([0,0,1.0])  # up in body frame

            e = np.cross(a / anorm, g_up_hat)
            bg -= ki*e*dt
            omega = gyro[k] - bg + kp*e

        # Integrate small correction
        q = quat_exp_omega(q, omega, dt)
        qs[k] = q

    qs[0] = qs[1]

    return qs


def augment_odometry_with_imu(traj_odom, twist_lin, imu_quats, R_imu_to_base=np.eye(3)):
    yaw_prev = None
    delta_ts = traj_odom.timestamps[1:] - traj_odom.timestamps[:-1]
    pos = np.zeros((len(traj_odom.timestamps), 3))

    assert len(twist_lin[1:]) == len(imu_quats[1:])
    assert len(twist_lin[1:]) == len(delta_ts)

    for idx, dt in enumerate(delta_ts):
        vl = twist_lin[idx+1]
        q = imu_quats[idx+1]

        R_w_b = tr.quaternion_matrix(q)[:3,:3] @ R_imu_to_base
        yaw = np.arctan2(R_w_b[1,0], R_w_b[0,0])

        if yaw_prev is None:
            yaw_prev = yaw

        # Get the mid yaw and wrap the angle
        # yaw_mid = 0.5 * (yaw_prev + yaw)
        yaw_mid = yaw
        yaw_mid = (yaw_mid + np.pi) % (2*np.pi) - np.pi

        v = vl[0]
        dx = v * np.cos(yaw_mid) * dt
        dy = v * np.sin(yaw_mid) * dt

        pos[idx+1] = pos[idx] + np.array([dx, dy, 0.])
        yaw_prev = yaw

    return PoseTrajectory3D(pos[1:], imu_quats[1:], traj_odom.timestamps[1:])


def gravity_from_attitude(qs):
    return np.array([
        # Gravity in body frame (down)
        tr.quaternion_matrix(q)[:3,:3].T @ np.array([0,0,-1])
        for q in qs
    ])


def estimate_gravity(acc, gyro, dt, g=9.80665, kp=1.0, ki=0.3):
    qs = mahony_filter(acc, gyro, dt, g, kp, ki)
    return gravity_from_attitude(qs)


if __name__ == "__main__":
    from fomo_utils import get_trajectory_dir, get_imu_data, get_transforms
    # Load data
    trajectory_dir = get_trajectory_dir()
    transform_manager = get_transforms(trajectory_dir)

    accel, gyro, timestamps = get_imu_data(trajectory_dir, imu='vectornav', tm=transform_manager)

    # Estimate the gravity vector
    freq = 200.0
    dt = 1/freq
    g_body = estimate_gravity(accel, gyro, dt, g=9.80665, kp=2.0, ki=0.05)
    assert len(g_body) == len(accel)
