import copy
from evo.core import transformations as tr
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


def mahony_filter(acc, gyro, dt, g=9.80665, kp=1.0, ki=0.3):
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

        # Use accel if credible
        a = acc[k]
        anorm = np.linalg.norm(a)
        omega = gyro[k]

        if 0.9*g < anorm < 1.1*g and np.linalg.norm(gyro[k]) < 3.0:
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
