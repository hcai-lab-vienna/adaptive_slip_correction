import pandas as pd
from fontTools.unicodedata import block
from statsmodels.tsa.api import VAR

pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import time
import tracemalloc
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from PH import PageHinkley
import os
import itertools
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from evo.core import  sync
from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D

from adaptive_slip_correction.fomo_utils import (
    DEPLOYMENTS,
    TRAJECTORIES,
    get_trajectory_dir,
    get_transforms,
    get_odom_trajectory,
    get_robot_cmd,
    get_gt_trajectory,
    get_imu_data
)
from adaptive_slip_correction.imu_utils import mahony_filter, gravity_from_attitude, augment_odometry_with_imu
from adaptive_slip_correction.trajectory_utils import (
    integrate_body_twists,
    compute_rpe_from_rel_pose,
    sync,
    reduce_to_ids,
    relative_pose_from_trajectories,
    velocities_from_deltaT
)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "..\\..\\..\\OSR\\code\\fomo-dataset"
NEW_FOMO_PATH = "..\\..\\..\\OSR\\code\\New_fomo_DATASET\\"

date_train = {"2025-11-03","2024-11-28",'2025-05-28',"2024-11-21","2025-09-24", "2025-10-14"}#"2025-06-26",
date_test = { "2025-04-15","2025-08-20"}
date_RAIN_road_test = {"2025-11-03"}
CONDITIONS_DATE=[ ("2025-04-15",1),("2025-11-03",3),("2024-11-28",1),("2025-06-26",2),("2025-09-24",2),("2024-11-21",2),("2025-05-28",2),("2025-08-20",2),("2025-10-14",2)]
CONDITION_CLASSIFICATION_EXPLAINATION=[(1,'snow on the road, not snowing'),
                          (2,'clear road, not raining'),
                          (3,'clear road, raining')]
CONDITION_CLASSIFICATION=[(1,'snow_road'),
                          (2,'clear_road'),
                          (3,'clear_raining')]

SV_MAX=[43,188,57]
LIN_VEL_MAX=[59,83,47]
SV_MAX=[44,161,152]
LIN_VEL_MAX=[91,144,112]

def build_dataframe(indices):
    df = pd.DataFrame(index=indices)
    return df

def fill_soil_Condition(df):
    # make sure datetime in index
    df.index = pd.to_datetime(df.index)
    #  array to dic
    condition_dict = {
        pd.to_datetime(date): value
        for date, value in CONDITIONS_DATE
    }
    '''drivetrain_dict = {
        pd.to_datetime(date): value
        for date, value in DRIVETRAIN_DATE
    }'''
    # create the column
    df["Soil_type"] = pd.Series(df.index.date,index=df.index).map(condition_dict)
    #df["Drivetrain_type"] = pd.Series(df.index.date,index=df.index).map(drivetrain_dict)
    return df

def get_odom_trajectoryMA(trajectory_dir,file):
    return file_interface.read_tum_trajectory_file(os.path.join(trajectory_dir, file))

def change_traj(traj, n):
    new_traj = PoseTrajectory3D(
        positions_xyz=traj.positions_xyz[n:],
        orientations_quat_wxyz=traj.orientations_quat_wxyz[n:],
        timestamps=traj.timestamps[n:]
    )

    return new_traj
# -------------------------------------------------
# Function to evaluate  T-kan SGDRegressor PassiveAgressiveRegressor
# -------------------------------------------------
def evaluate_modelPH(name, model,X_train, y_train,X_test,y_test, flag,online=False):
    print(f"\n===== {name} =====")
    '''
    flag=0      SGDRegressor
    flag = 1    SGDPassive Agressive
    flag = 2    SGDRegressor
    '''
    tracemalloc.start()
    start_train = time.time()

    train_time = time.time() - start_train
    start_test = time.time()

    preds = []

    if flag==0:
        ph = PageHinkley(delta=0.0019, lambda_=26,alpha=0.999)
    elif flag==1:
        ph = PageHinkley(delta=0.002, lambda_=20,alpha=0.999)
    else:
        ph = PageHinkley(delta=0.1, lambda_=150, alpha=0.95)

    drifts = []
    cooldown = 100
    last_drift = -cooldown

    buffer_X = []
    buffer_y = []
    window = 20

    for i in range(len(X_test)):
        x = X_test[i].reshape(1, -1)
        y_true = y_test[i]

        if name=='T-KAN (real)':
            y_pred = float(model.predict(x))
            preds.append(y_pred)
        else:
            y_pred = float((model.predict(x))[0])
            preds.append(y_pred)

        if online:
            buffer_X.append(x)
            buffer_y.append(y_true)

            if len(buffer_X) > window:
                buffer_X.pop(0)
                buffer_y.pop(0)
            model.partial_fit(np.vstack(buffer_X), buffer_y)

        error = y_true - y_pred
        drift = ph.update(error)

        if drift and (i - last_drift) >= cooldown:
            drifts.append(i)
            last_drift = i

    test_time = time.time() - start_test
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
    print("Train time:", train_time)
    print("Test time:", test_time)
    print("Peak memory (KB):", peak / 1024)
    print("Drifts detected:", drifts)

    '''plt.figure(figsize=(12, 5))
    plt.title(f"{name}")
    plt.scatter(np.arange(len(y_test)), y_test, label="Real", color='r', linewidth=1,s=1)
    plt.scatter(np.arange(len(y_test)), preds, label="Prediction", color='b',linewidth=1,s=1)

    for d in drifts:
        plt.axvline(x=d, color='black', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.show(block=True)'''

    return {
        "model": name,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "train_time": train_time,
        "test_time": test_time,
        "memory_kb": peak / 1024,
        "drifts": drifts
    },preds

def create_lags_fast(X, y, lags=10):
    X_lagged = []
    y_lagged = []
    for i in range(lags, len(y)):
        X_lagged.append(
            np.hstack((X[i], y[i-lags:i]))
        )
        y_lagged.append(y[i])
    return np.array(X_lagged), np.array(y_lagged)

def create_sequences_flat(X, y, lags=15):
    X_seq = []
    y_seq = []

    for i in range(lags, len(X)):
        X_seq.append(X[i - lags:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_seq = X_seq.reshape(X_seq.shape[0], -1)

    return X_seq, y_seq

# ============================================================
# LOAD and PREPROCESS FUNCTIONS
# ============================================================
def load_meta(path,season):
    print(path)
    # ---------- meteo ----------
    file_meteo = f"{path}\meteo_data.csv"
    if os.path.exists(file_meteo) and os.path.getsize(file_meteo) > 0:
        meteo = pd.read_csv(file_meteo)
        meteo.drop(0, axis=0, inplace=True)#units
        meteo.drop(1, axis=0, inplace=True)#NAN
        meteo.drop('RECORD', axis=1, inplace=True)
        meteo["TIMESTAMP"]  = meteo["TIMESTAMP"] .astype(float)

        meteo = meteo.drop_duplicates()
        meteo = meteo.sort_values("TIMESTAMP")
        meteo = meteo.set_index("TIMESTAMP")
        meteo_all_zero = meteo.select_dtypes(include="number").columns[
            (meteo.select_dtypes(include="number") == 0).all()
        ]
        meteo = meteo.loc[(meteo != 0).any(axis=1)]
        METEO_COLUMNS=['Rain_mm_Tot', 'Rain_accumulation','T_probe_Avg','RH_probe','T_DP_Probe','CS106_Corrected_mbar']
        METEO_COLUMNS=['Rain_accumulation','T_probe_Avg','RH_probe','T_DP_Probe','CS106_Corrected_mbar']
        meteo=meteo[METEO_COLUMNS]
        meteo = meteo.rename(columns={'RH_probe': 'RH', 'Rain_accumulation': 'Rain_accum'})
        meteo = meteo.apply(pd.to_numeric, errors="coerce")
    else:
        print(f"⚠️ Empty file: {file_meteo}")
        file_meteo = f"{path}\meteo_data.dat"

        meteo = pd.read_csv(
            file_meteo,
            header=1,
            skiprows=[2, 3],
            parse_dates=["TIMESTAMP"]
        )
        meteo.drop('RECORD', axis=1, inplace=True)
        meteo = meteo.drop_duplicates()
        meteo = meteo.set_index("TIMESTAMP")

        meteo_all_zero = meteo.select_dtypes(include="number").columns[
            (meteo.select_dtypes(include="number") == 0).all()
        ]

        meteo = meteo.loc[(meteo != 0).any(axis=1)]
        METEO_COLUMNS = ['Rain_mm_Tot', 'Rain_accumulation', 'T_probe_Avg', 'RH_probe', 'T_DP_Probe',
                         'CS106_Corrected_mbar']
        METEO_COLUMNS = ['Rain_accumulation', 'T_probe_Avg', 'RH_probe', 'T_DP_Probe', 'CS106_Corrected_mbar']
        meteo = meteo[METEO_COLUMNS]
        meteo = meteo.rename(columns={'RH_probe': 'RH', 'Rain_accumulation': 'Rain_accum'})
    # ---------- snow ----------
    file_path=f"{path}\snow_data.csv"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        snow = pd.read_csv(file_path )
        snow.drop(0, axis=0, inplace=True)  # units
        snow.drop(1, axis=0, inplace=True) #NAN
        snow.drop('RECORD', axis=1, inplace=True)
        snow = snow.apply(pd.to_numeric, errors="coerce")
        snow["TIMESTAMP"]=snow["TIMESTAMP"].astype(float)
        snow = snow.drop_duplicates()
        snow = snow.sort_values("TIMESTAMP")
        snow = snow.set_index("TIMESTAMP")
        snow_all_zero = snow.select_dtypes(include="number").columns[
            (snow.select_dtypes(include="number") == 0).all()
        ]
        cols = [c for c in snow.columns if str(c).startswith("SDMS40_Distance_Points")]
        snow["SDMS40_Distance_Avg"] = snow[cols].mean(axis=1)
        SNOW_COLUMNS = ['SDMS40_Depth_Avg', 'SDMS40_Board_Temperature',
           'SDMS40_Heater_Low_Threshold_Temperature', 'SDMS40_Laser_Temperature','SDMS40_Distance_Avg']
        SNOW_COLUMNS = ['SDMS40_Depth_Avg']
        snow = snow[SNOW_COLUMNS]
        snow = snow.rename(columns={'SDMS40_Depth_Avg': 'SnowDepth_Avg'})
    else:
        print(f"⚠️ Empty file: {file_path}")
        file_path = f"{path}\snow_data.dat"
        snow = pd.read_csv(
            file_path,
            header=1,
            skiprows=[2, 3],
            parse_dates=["TIMESTAMP"]
        )
        snow.drop('RECORD', axis=1, inplace=True)
        snow = snow.drop_duplicates()
        snow = snow.set_index("TIMESTAMP")
        snow_all_zero = snow.select_dtypes(include="number").columns[
            (snow.select_dtypes(include="number") == 0).all()
        ]
        cols = [c for c in snow.columns if str(c).startswith("SDMS40_Distance_Points")]
        snow["SDMS40_Distance_Avg"] = snow[cols].mean(axis=1)
        SNOW_COLUMNS = ['SDMS40_Depth_Avg', 'SDMS40_Board_Temperature',
                        'SDMS40_Heater_Low_Threshold_Temperature', 'SDMS40_Laser_Temperature',
                        'SDMS40_Distance_Avg']
        SNOW_COLUMNS = ['SDMS40_Depth_Avg']
        snow = snow[SNOW_COLUMNS]
        snow = snow.rename(columns={'SDMS40_Depth_Avg': 'SnowDepth_Avg'})

    dftmp = [meteo, snow]
    dfs_validostmp = []
    for df in dftmp:
        if df is not None and not df.empty:
            dfs_validostmp.append(df)
        else:
            print(" Empty DataFrame")
    flag_INTERSECTION=0
    if len(dfs_validostmp) != 0:
        for df in dfs_validostmp:
            df.index = pd.to_datetime(df.index, unit='us')
            print(df.index.min(), df.index.max())
        interseccion = dfs_validostmp[0].index.intersection(dfs_validostmp[1].index)

        if interseccion.empty:
            print("No intersection")
            flag_INTERSECTION=1
        else:
            print("There is intersection")

            start = max(df.index.min() for df in dfs_validostmp)
            end = min(df.index.max() for df in dfs_validostmp)

            dfs_validostmp = [df.loc[start:end] for df in dfs_validostmp]
        tmp_index = dfs_validostmp[0].index

        # ---------- TERRAIN ----------
        df_terreno = build_dataframe(tmp_index)
        df_terreno = fill_soil_Condition(df_terreno)

        name_file=str('Terrain_')+str(season)+str('.csv')
        df_terreno.to_csv(f"{path}/{name_file}",sep=',', header=True,index=True)
    else:
        print(f"⚠️ EMPTY DATAFRAME.")
        df_terreno = pd.DataFrame()

    # ---------- MERGE ----------
    dfmeteo = [meteo, snow, df_terreno]
    dfs_validos = []
    for df in dfmeteo:
        if df is not None and not df.empty:
            dfs_validos.append(df)
        else:
            print(" Empty DataFrame")

    if len(dfs_validostmp) != 0:
        if flag_INTERSECTION!=0:
            first_valid_value = snow['SnowDepth_Avg'].iloc[0]
            master_index = dfs_validostmp[0].index
            aligned = []
            for df in dfmeteo:
                df_interp = (
                    df.reindex(master_index)
                )
                aligned.append(df_interp)

            df_meteo_final = pd.concat(aligned, axis=1)
            df_meteo_final['SnowDepth_Avg']=first_valid_value
        else:
            master_index = dfs_validostmp[0].index
            aligned = []
            for df in dfmeteo:
                df_interp = (
                    df.reindex(master_index)
                )
                aligned.append(df_interp)
            df_meteo_final = pd.concat(aligned, axis=1)
    else:
        print(f"⚠️ Final DATAFRAME:EMPTY.")
        df_meteo_final = pd.DataFrame()

    return df_meteo_final

def load_trajectory_data2(fecha, traj):
    # Load data
    trajectory_dir = get_trajectory_dir(deployment=fecha, trajectory=traj)
    # Transforms
    transform_manager = get_transforms(trajectory_dir)
    # IMUs
    # imu_name = 'vectornav'
    imu_name = 'xsens'
    accel, gyro, imu_timestamps = get_imu_data(trajectory_dir, imu=imu_name, tm=transform_manager)
    # Trajectories
    traj_gt = get_gt_trajectory(trajectory_dir)
    traj_odom, lin_vel_twist, ang_vel_twist = get_odom_trajectory(trajectory_dir)
    cmd_df = get_robot_cmd(trajectory_dir, return_df=True)

    # Compute properties
    # IMUs
    g_earth = 9.80665
    imu_quats = mahony_filter(accel, gyro, imu_timestamps, g=g_earth, kp=1.0, ki=0.3)
    g_body = gravity_from_attitude(imu_quats)
    accel_grav_compensated = accel + g_earth * g_body
    # GT trajectory
    delta_ts_gt = traj_gt.timestamps[1:] - traj_gt.timestamps[:-1]
    p_rel_gt = relative_pose_from_trajectories([traj_gt])[0]
    vel_gt = np.linalg.norm(np.array(p_rel_gt)[:, :3, 3], axis=1) / delta_ts_gt

    gt_timestamps = traj_gt.timestamps[1:]
    # Wheel + IMU Odometry
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

    # Align start-end and interpolate
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
    gt_df = pd.DataFrame({
        'ts': pd.to_datetime(1e9 * gt_timestamps),
        'vel_gt': vel_gt,
        'x_gt': traj_gt.positions_xyz[1:, 0],
        'y_gt': traj_gt.positions_xyz[1:, 1],
        'z_gt': traj_gt.positions_xyz[1:, 2],
    })

    idice_gt = pd.DataFrame({'raw_index': pd.to_datetime(1e9 * gt_timestamps)})
    gt_df = gt_df.set_index('ts')

    vel_gt_sync = gt_df['vel_gt'].reindex(odom_dt_index, method='nearest', tolerance=pd.Timedelta('50ms')).interpolate(method='time').values
    pos_gt_sync = gt_df[['x_gt', 'y_gt', 'z_gt']].reindex(odom_dt_index, method='nearest',tolerance=pd.Timedelta('20ms')).interpolate( method="time").values

    traj_gt_sync = PoseTrajectory3D(
        pos_gt_sync,
        np.array([[1.0, 0., 0., 0.]] * len(odom_dt_index)),
        traj_odom_imu.timestamps[1:])

    p_rel_gt_sync, p_rel_odom_imu_sync = relative_pose_from_trajectories([traj_gt_sync, traj_odom_imu])
    odom_rpe = compute_rpe_from_rel_pose(p_rel_gt_sync, p_rel_odom_imu_sync)
    print("odom_rpe", odom_rpe.mean(), odom_rpe.min(), odom_rpe.max(), odom_rpe.shape)

    DATASET = pd.DataFrame(
        np.hstack([
            accel_grav_compensated_sync[:, :2],
            gyro_sync[:, -1].reshape(-1, 1),
            g_body_sync[:, 0].reshape(-1, 1),
            np.array(lin_vel_odom_imu)[:, 0].reshape(-1, 1),
            np.array(ang_vel_odom_imu)[:, -1].reshape(-1, 1),
            lin_vel_cmd[:, 0].reshape(-1, 1),
            ang_vel_cmd[:, -1].reshape(-1, 1),
            vel_gt_sync.reshape(-1, 1)
        ]),
        columns=[
            'lin_acc_imu_x',
            'lin_acc_imu_y',
            'ang_vel_imu_z',
            'grav_x',
            'lin_vel_odom_x',
            'ang_vel_odom_z',
            'lin_vel_cmd_x',
            'ang_vel_cmd_z',
            'TARGET'
        ]
    )
    if len(odom_dt_index) != DATASET.shape[0]:
        print('DIMMENSIONAL ERROR')

    DATASET = DATASET.set_index(odom_dt_index)
    DATASET = DATASET.dropna()

    print('Dimensions DATASET ', DATASET.shape)
    print('Dimensions odom_POSITIONS ', traj_odom_imu.positions_xyz.shape)
    print('Dimensions gt_POSITIONS ', traj_gt.positions_xyz.shape)

    # plot_trajectories(fecha,traj_odom_imu, traj_odom, traj_gt)

    return DATASET, traj_odom_imu.positions_xyz[1:], odom_dt_index, idice_gt, traj_gt_sync, odom_rpe

def plot_trajectories(fecha,traj_odom_imu,traj_odom,traj_gt):
    plt.figure(figsize=(12, 5))
    plt.title(str(fecha)+" PLOT trajectories")

    plt.scatter(traj_odom_imu.positions_xyz[0, 0], traj_odom_imu.positions_xyz[0, 1],alpha=0.4,color='g',marker='o')
    plt.scatter(traj_odom_imu.positions_xyz[-1,0], traj_odom_imu.positions_xyz[-1,1],alpha=0.4,color='g',marker='o')
    plt.plot(traj_odom_imu.positions_xyz[:, 0], traj_odom_imu.positions_xyz[:, 1],linestyle=':',color='g',label='Correction')

    plt.scatter(traj_odom.positions_xyz[0, 0], traj_odom.positions_xyz[0, 1], alpha=0.4,color='b', marker='s')
    plt.scatter(traj_odom.positions_xyz[-1, 0], traj_odom.positions_xyz[-1, 1], alpha=0.4,color='b', marker='s')
    plt.plot(traj_odom.positions_xyz[:, 0], traj_odom.positions_xyz[:, 1],linestyle='--',color='b',label='Pose')

    plt.scatter(traj_gt.positions_xyz[1, 0]- traj_gt.positions_xyz[0, 0], traj_gt.positions_xyz[1, 1]- traj_gt.positions_xyz[0, 1], alpha=0.4,color='r', marker='X')
    plt.scatter(traj_gt.positions_xyz[-1, 0]- traj_gt.positions_xyz[0, 0], traj_gt.positions_xyz[-1, 1]- traj_gt.positions_xyz[0, 1], alpha=0.4,color='r', marker='X')
    plt.plot(traj_gt.positions_xyz[:, 0] - traj_gt.positions_xyz[0, 0],
             traj_gt.positions_xyz[:, 1] - traj_gt.positions_xyz[0, 1],color='r',label='Ground Truth')
    plt.legend()
    plt.show(block=True)

def FEATURE_IMP(input,flag_trin_test, features,taget_lab):
    print("\nTraining independent models...")

    if 1 in input:
        df_snow_road = input[1]
    else:
        df_snow_road=pd.DataFrame()
        model_snow_road = None
    if 2 in input:
        df_clear_road = input[2]
    else:
        df_clear_road=pd.DataFrame()
        model_clear_road = None
    if 3 in input:
        df_clear_raining = input[3]
    else:
        df_clear_raining=pd.DataFrame()
        model_clear_raining = None

    if not df_snow_road.empty:
        model_snow_road = train_model(df_snow_road,features,taget_lab)
    if not df_clear_road.empty:
        model_clear_road = train_model(df_clear_road,features,taget_lab)
    if not df_clear_raining.empty:
        model_clear_raining = train_model(df_clear_raining,features,taget_lab)

    models = {
        "snow_road": (df_snow_road, model_snow_road),
        "clear_road": (df_clear_road, model_clear_road),
        "clear_raining": (df_clear_raining, model_clear_raining),
    }

    fi_dict = {
        name: model.feature_importances_
        for name, (df, model) in models.items()
        if not df.empty
    }

    fi = pd.DataFrame(fi_dict, index=features)

    '''fi = pd.DataFrame(
        {
            "snow_road": model_snow_road.feature_importances_,
            "snow_snowing": model_snow_snowing.feature_importances_,
            "clear_road": model_clear_road.feature_importances_,
            "clear_raining": model_clear_raining.feature_importances_,
        },
        index=features
    )'''
    print(features)
    print("\nFeature Importance:")
    print(fi)
    if flag_trin_test==0:
        name=' TRAIN'
    else:
        name = ' TEST'

    title_plt=[ "Changes in the relationship between SV "+str(name) ]

    '''fi.plot(kind="bar", title=title_plt[0],color=["gold", "blue","green","orange"],fontsize=18)
    plt.ylabel("Feature importance",fontsize=16)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=9)
    plt.show(block=True)'''

def plot_cdf(data, label):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, yvals, label=label)

def treatment_XGB_season(input,test,features,LABEL_SV):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    df_snow_road = df_snow_road.copy()
    df_clear_road = df_clear_road.copy()
    df_clear_raining = df_clear_raining.copy()

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL_SV].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL_SV].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL_SV].values

    mediasnow = np.mean(y_snow_road)
    medianasnow = np.median(y_snow_road)
    mediaclear = np.mean(y_clear_road)
    medianaclear = np.median(y_clear_road)
    mediarain = np.mean(y_clear_raining)
    medianarain = np.median(y_clear_raining)

    print('SNOW mean & median ',mediasnow,medianasnow)
    print('CLEAR mean & median ', mediaclear, medianaclear)
    print('RAIN mean & median ', mediarain, medianarain)

    corr_snow = df_snow_road.corr(numeric_only=True)
    corr_clear = df_clear_road.corr(numeric_only=True)
    corr_rain = df_clear_raining.corr(numeric_only=True)
    mask_snow = np.abs(corr_snow) < 0.5
    mask_clear = np.abs(corr_clear) < 0.5
    mask_rain = np.abs(corr_rain) < 0.5

    '''print(corr_rain.shape)
    print(mask_rain.shape)
    print(corr_rain.index.equals(pd.Index(mask_rain.index)))

    print(corr_clear.shape)
    print(mask_clear.shape)
    print(corr_clear.index.equals(pd.Index(mask_clear.index)))

    print(corr_snow.shape)
    print(mask_snow.shape)
    print(corr_snow.index.equals(pd.Index(mask_snow.index)))'''

    corr_check = corr_rain.copy()
    corr_check.insert(0, "feature_name", corr_check.index)
    #print(corr_check["feature_name"])

    corr_check2 = corr_clear.copy()
    corr_check2.insert(0, "feature_name", corr_check2.index)
    #print(corr_check2["feature_name"])

    corr_check3 = corr_snow.copy()
    corr_check3.insert(0, "feature_name", corr_check3.index)
    #print(corr_check3["feature_name"])

    corr_target_snow = corr_snow[LABEL_SV].dropna()
    corr_target_clear = corr_clear[LABEL_SV].dropna()
    corr_target_rain = corr_rain[LABEL_SV].dropna()

    ks_snow_rain = ks_2samp(df_snow_road[LABEL_SV], df_clear_raining[LABEL_SV])
    ks_snow_clear = ks_2samp(df_snow_road[LABEL_SV], df_clear_road[LABEL_SV])
    ks_clear_rain = ks_2samp(df_clear_road[LABEL_SV], df_clear_raining[LABEL_SV])

    print('snow-rain: Kolmogorov Smirnov Test Results:' , ks_snow_rain)
    print('snow-clear: Kolmogorov Smirnov Test Results:' , ks_snow_clear)
    print('clear-rain: Kolmogorov Smirnov Test Results:' , ks_clear_rain)

    plt.figure(figsize=(10, 8))
    plot_cdf(df_snow_road[LABEL_SV], "snow")
    plot_cdf(df_clear_road[LABEL_SV], "clear")
    plot_cdf(df_clear_raining[LABEL_SV], "rain")
    plt.legend()
    plt.title("CDF comparison")
    plt.xlabel(LABEL_SV)
    plt.ylabel("F(x)")
    plt.grid()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].set_title("Histogram")
    sns.histplot(y_snow_road, alpha=0.5, bins=1000, color='r', label='snow',ax=axes[0])
    axes[0].axvline(mediasnow, color='red', linestyle='solid', linewidth=2, label=f'Mean snow: {mediasnow:.2f}')
    axes[0].axvline(medianasnow, color='red', linestyle='dashed', linewidth=2, label=f'Median snow: {medianasnow:.2f}')

    sns.histplot(y_clear_road, alpha=0.5, bins=1000, color='g', label='clear',ax=axes[0])
    axes[0].axvline(mediaclear, color='k', linestyle='solid', linewidth=2, label=f'Mean clear: {mediaclear:.2f}')
    axes[0].axvline(medianaclear, color='k', linestyle='dashed', linewidth=2, label=f'Median clear: {medianaclear:.2f}')

    sns.histplot(y_clear_raining, alpha=0.5, bins=1000, color='b', label='rain',ax=axes[0])
    axes[0].axvline(mediarain, color='yellow', linestyle='solid', linewidth=2, label=f'Mean rain: {mediarain:.2f}')
    axes[0].axvline(medianarain, color='yellow', linestyle='dashed', linewidth=2, label=f'Median rain: {medianarain:.2f}')
    axes[0].legend()
    axes[0].set_title(" SV Histograms")

    sns.kdeplot(df_snow_road[LABEL_SV], label="snow", color='r',fill=True,ax=axes[1])
    sns.kdeplot(df_clear_road[LABEL_SV], label="clear", color='g',fill=True,ax=axes[1])
    sns.kdeplot(df_clear_raining[LABEL_SV], label="rain", color='b',fill=True,ax=axes[1])
    axes[1].legend()
    axes[1].set_title("KDE Distributions")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.125,
        bottom=0.223,
        right=1,
        top=0.962,
        wspace=0.2,
        hspace=0.2
    )
    sns.heatmap(
        corr_clear,
        mask=mask_clear,
        annot=True,
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True
    )
    plt.yticks(rotation=0)
    plt.title("CLEAR ROAD - Correlations ")
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.125,
        bottom=0.223,
        right=1,
        top=0.962,
        wspace=0.2,
        hspace=0.2
    )
    sns.heatmap(
        corr_snow,
        mask=mask_snow,
        annot=True,
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True
    )
    plt.yticks(rotation=0)
    plt.title("SNOW - Correlations ")
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.125,
        bottom=0.223,
        right=1,
        top=0.962,
        wspace=0.2,
        hspace=0.2
    )
    sns.heatmap(
        corr_rain,
        mask=mask_rain,
        annot=True,
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True
    )
    plt.yticks(rotation=0)
    plt.title("RAIN - Correlations ")
    plt.show()

    #plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.158,
        bottom=0.223,
        right=1,
        top=0.926,
        wspace=0.2,
        hspace=0.2
    )
    sns.heatmap(
        corr_target_snow.to_frame(name="correlation"),
        annot=True,
        cmap="coolwarm",
        center=0,
        yticklabels=True,
        xticklabels=True
    )
    plt.title("SV - SNOW  - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.158,
        bottom=0.223,
        right=1,
        top=0.926,
        wspace=0.2,
        hspace=0.2
    )
    sns.heatmap(
        corr_target_rain.to_frame(name="correlation"),
        annot=True,
        cmap="coolwarm",
        center=0,
        yticklabels=True,
        xticklabels=True
    )
    plt.title("SV- RAIN - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.158,
        bottom=0.223,
        right=1,
        top=0.926,
        wspace=0.2,
        hspace=0.2
    )
    sns.heatmap(
        corr_target_clear.to_frame(name="correlation"),
        annot=True,
        cmap="coolwarm",
        center=0,
        yticklabels=True,
        xticklabels=True
    )
    plt.title("SV- CLEAR - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show()

    df_all = pd.concat([
        df_snow_road,
        df_clear_road,
        df_clear_raining
    ])

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.125,
        bottom=0.223,
        right=1,
        top=0.962,
        wspace=0.2,
        hspace=0.2
    )
    sns.boxplot(data=df_all, x="Soil_type", y=LABEL_SV)
    plt.title("SV distribution by condition")
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.125,
        bottom=0.223,
        right=1,
        top=0.962,
        wspace=0.2,
        hspace=0.2
    )
    sns.boxplot(data=df_all, x="Soil_type", y=LABEL_SV)
    sns.stripplot(data=df_all, x="Soil_type", y=LABEL_SV,
                  color="black", alpha=0.3)
    plt.title("Boxplot real distribution")
    plt.show()

    fig, ax = plt.subplots()
    fig.subplots_adjust(
        left=0.125,
        bottom=0.223,
        right=1,
        top=0.962,
        wspace=0.2,
        hspace=0.2
    )
    sns.violinplot(data=df_all, x="Soil_type", y=LABEL_SV)
    plt.title("Kernel density–based distribution of SV by condition.")# SV distribution by  condition
    plt.show()

    summary = df_all.groupby("Soil_type")[LABEL_SV].describe()
    print(summary)

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index=df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road[LABEL_SV].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road[LABEL_SV].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining[LABEL_SV].values

    ks_snow_train_test = ks_2samp(df_snow_road[LABEL_SV], df_TEST_snow_road[LABEL_SV])
    ks_rain_train_test = ks_2samp(df_clear_raining[LABEL_SV], df_TEST_clear_raining[LABEL_SV])
    ks_clear_train_test = ks_2samp(df_clear_road[LABEL_SV], df_TEST_clear_road[LABEL_SV])

    print('SNOW-train-test: Kolmogorov Smirnov Test Results:', ks_snow_train_test)
    print('RAIN-train-test: Kolmogorov Smirnov Test Results:', ks_rain_train_test)
    print('CLEAR-train-test: Kolmogorov Smirnov Test Results:', ks_clear_train_test)


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].set_title("Histograma TRAIN TEST SNOW")
    sns.histplot(y_snow_road, alpha=0.5, bins=1000, color='b', label='snowTRAIN', ax=axes[0])
    sns.histplot(y_TEST_snow_road, alpha=0.5, bins=1000, color='r', label='snow TEST', ax=axes[0])
    axes[0].legend()
    axes[0].set_title(" SV TRAIN TEST Histograms")

    sns.kdeplot(df_snow_road[LABEL_SV], label="snow TRAIN", color='b', fill=True, ax=axes[1])
    sns.kdeplot(df_TEST_snow_road[LABEL_SV], label="snow TEST", color='r', fill=True, ax=axes[1])
    axes[1].legend()
    axes[1].set_title("KDE TRAIN TEST Distributions")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].set_title("Histograma TRAIN TEST CLEAR")
    sns.histplot(y_clear_road, alpha=0.5, bins=1000, color='b', label='clear TRAIN', ax=axes[0])
    sns.histplot(y_TEST_clear_road, alpha=0.5, bins=1000, color='r', label='clear TEST', ax=axes[0])
    axes[0].legend()
    axes[0].set_title(" SV TRAIN TEST Histograms")

    sns.kdeplot(df_clear_road[LABEL_SV], label="clear TRAIN", color='b', fill=True, ax=axes[1])
    sns.kdeplot(df_TEST_clear_raining[LABEL_SV], label="clear TEST", color='r', fill=True, ax=axes[1])
    axes[1].legend()
    axes[1].set_title("KDE TRAIN TEST Distributions")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].set_title("Histograma TRAIN TEST RAIN")
    sns.histplot(y_clear_raining, alpha=0.5, bins=1000, color='b', label='rain RAIN', ax=axes[0])
    sns.histplot(y_TEST_clear_raining, alpha=0.5, bins=1000, color='r', label='rain TEST', ax=axes[0])
    axes[0].legend()
    axes[0].set_title(" SV TRAIN TEST Histograms")

    sns.kdeplot(df_clear_raining[LABEL_SV], label="rainTRAIN", color='b', fill=True, ax=axes[1])
    sns.kdeplot(df_TEST_clear_raining[LABEL_SV], label="rain TEST", color='r', fill=True, ax=axes[1])
    axes[1].legend()
    axes[1].set_title("KDE TRAIN TEST Distributions")
    plt.tight_layout()
    plt.show()

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]
    LABELS_ENTRENAMIENTO = [y_snow_road, y_clear_road, y_clear_raining]
    INDICES = [y_test_snow_road_index, y_test_clear_road_index, y_test_clear_raining_index]
    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]


    results = []
    prediction = []
    conditions = ["snow_road", "clear_road", "clear_raining"]
    for est1, X_train, y_train, X_test, y_test in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training conditions: {est1}")

        modelxgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        modelxgb.fit(X_train, y_train)

        '''importance = modelxgb.feature_importances_
        df_xgb = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values(by="importance", ascending=False)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.barh(df_xgb["feature"], df_xgb["importance"])
        plt.gca().invert_yaxis()
        plt.title(str(est1)+" Feature Importance - XGBoost")
        plt.show()'''

        contador=0
        for est2, X_test, y_test in zip(
                conditions,
                DATOS_TEST,
                LABELS_TEST
        ):
            print(f"Prediction conditions: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            prediction_xgb=modelxgb.predict(X_test)
            prediction.append(prediction_xgb)

            mae_pred = mean_absolute_error(y_test, prediction_xgb)
            r2=r2_score(y_test, prediction_xgb)

            print(f"Time series: Real {est1} vs Prediction {est2}")
            print(f"MAE : {mae_pred:.4f}")

            '''plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter(np.arange(0, len(y_test)), prediction_xgb, label="Prediction XGB", color='green',linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)'''

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                'mae':mae_pred,
                'r2':r2,
                'prediccion':prediction_xgb
            })
            contador+=1

    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)
    matriz_mae = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="mae"
    )

    matriz_r2 = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="r2"
    )

    print("Matrix MAE:")
    print(matriz_mae)

    print("\nMatrix R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("XGB MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("XGB R2 (train vs test)")
    plt.show()

def treatment_SGDseason(input,test,features,LABEL_SV):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL_SV].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL_SV].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL_SV].values

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index = df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road[LABEL_SV].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road[LABEL_SV].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]#, X_clear_raining
    LABELS_ENTRENAMIENTO = [y_snow_road,  y_clear_road, y_clear_raining]#, y_clear_raining
    INDICES = [y_test_snow_road_index,  y_test_clear_road_index, y_test_clear_raining_index ]#, y_test_clear_raining_index
    DATOS_TEST = [X_TEST_snow_road,  X_TEST_clear_road, X_TEST_clear_raining]#, X_TEST_clear_raining
    LABELS_TEST = [y_TEST_snow_road,  y_TEST_clear_road, y_TEST_clear_raining]#, y_TEST_clear_raining

    results = []
    prediction = []
    coeficients = {}
    conditions = ["snow_road",  "clear_road", "clear_raining"]#, "clear_raining"
    for est1, X_train, y_train, X_test, y_test in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training Conditions: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        lag = len(features)

        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        # SGD
        sgd1 = SGDRegressor(
            max_iter=2000,
            loss='epsilon_insensitive',  # Passive Aggressive Esto actúa como SVR online.
            learning_rate='pa1',
            eta0=1,
            alpha=0.001,
            shuffle=False,
            random_state=42,
            penalty=None
        )

        sgd1.fit(X_train, y_train)
        #coefs = np.abs(sgd1.coef_)  # shape (32,)
        # reshape → (n_features, n_lags)
        #coefs_reshaped = coefs.reshape(len(features), 2)
        #importance_per_feature = coefs_reshaped.sum(axis=1)

        coeficients[est1] = {
            "coef": sgd1.coef_.copy(),
            "intercept": sgd1.intercept_.copy()
        }

        contador=0
        for est2, X_test, y_test in zip(
                conditions,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)  # create_lags
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Prediction Conditions: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            title_plt=f"Real {est1} vs Prediction {est2}"+ " SGDRegressor without PartialFit"
            resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH(title_plt, sgd1, X_train, y_train, X_test,
                                                                              y_test, 0,online=False)
            pr1=sgd1.predict(X_test)
            pr1inenversa = y_scaler.inverse_transform(np.array(pr1).reshape(-1, 1)).flatten()
            prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
                np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
            prediction.append(prediccion_sgd_offlinenversa)

            y_test_inverse = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
            mse_pred = mean_squared_error(y_test_inverse, prediccion_sgd_offlinenversa)
            mae_pred = mean_absolute_error(y_test_inverse, prediccion_sgd_offlinenversa)
            r2 = r2_score(y_test_inverse, prediccion_sgd_offlinenversa)

            '''df_sgd = pd.DataFrame({
                "feature": features,
                "importance": importance_per_feature
            }).sort_values(by="importance", ascending=False)

            values = df_sgd["importance"]
            colors = cm.viridis((values - values.min()) / (values.max() - values.min()))

            plt.figure(figsize=(8, 5))
            plt.barh(df_sgd["feature"], df_sgd["importance"],color=colors)
            plt.gca().invert_yaxis()
            plt.title(str(est1)+" Feature Importance - SGD")
            plt.show()'''

            plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter( np.arange(0, len(y_test)),prediccion_sgd_offlinenversa, label="Prediction SGD", color='turquoise', linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                'mae': mae_pred,
                'mse':mse_pred,
                'r2': r2,
                'prediccion': prediccion_sgd_offlinenversa
            })
            contador+=1
    for c1, c2 in itertools.combinations(conditions, 2):
        w1 = coeficients[c1]["coef"]
        w2 = coeficients[c2]["coef"]

        diff_norm = np.linalg.norm(w1 - w2)
        relative_diff = diff_norm / np.linalg.norm(w1)

        print(f"{c1} vs {c2}")
        print("Norm difference:", diff_norm)
        print("Relative difference:", relative_diff)
        print()
    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)

    matriz_mae = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="mae"
    )

    matriz_r2 = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="r2"
    )

    print("Matrix MAE:")
    print(matriz_mae)

    print("\nMatrix R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("SGD MAE(train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("SGD R2 (train vs test)")
    plt.show()

def treatment_elasticseason(input,test,features,LABEL_SV):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road =  10 *(df_snow_road[LABEL_SV] - df_snow_road['lin_vel_odom_x']).values#10 *
    X_clear_road = df_clear_road[features].values
    y_clear_road =  10 *(df_clear_road[LABEL_SV] - df_clear_road['lin_vel_odom_x']).values#10 *
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining =  10 *(df_clear_raining[LABEL_SV] - df_clear_raining['lin_vel_odom_x']).values#10 *

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index = df_TEST_snow_road.index
    y_TEST_snow_road =  10 *(df_TEST_snow_road[LABEL_SV]-df_TEST_snow_road['lin_vel_odom_x']).valuess#10 *
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road =  10 *(df_TEST_clear_road[LABEL_SV]-df_TEST_clear_road['lin_vel_odom_x']).valuess#10 *
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = 10 *(df_TEST_clear_raining[LABEL_SV]-df_TEST_clear_raining['lin_vel_odom_x']).valuess#10 *

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]#, X_clear_raining
    LABELS_ENTRENAMIENTO = [y_snow_road,  y_clear_road, y_clear_raining]#, y_clear_raining
    INDICES = [y_test_snow_road_index,  y_test_clear_road_index, y_test_clear_raining_index ]#, y_test_clear_raining_index
    DATOS_TEST = [X_TEST_snow_road,  X_TEST_clear_road, X_TEST_clear_raining]#, X_TEST_clear_raining
    LABELS_TEST = [y_TEST_snow_road,  y_TEST_clear_road, y_TEST_clear_raining]#, y_TEST_clear_raining

    results = []
    prediction = []
    conditions = ["snow_road",  "clear_road", "clear_raining"]#, "clear_raining"
    for est1, X_train, y_train, X_test, y_test in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training Conditions: {est1}")
        elas1 = ElasticNet(alpha=0.1,l1_ratio=0.2,max_iter=10000,
                tol=1e-5, random_state=42,selection='cyclic' )

        elas1.fit(X_train, y_train)

        contador=0
        for est2, X_test, y_test in zip(
                conditions,
                DATOS_TEST,
                LABELS_TEST
        ):
            print(f"Prediction Conditions: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            title_plt=f"Real {est1} vs Prediction {est2}"+ " SGDRegressor without PartialFit"
            pr1=elas1.predict(X_test)
            prediction.append(pr1)

            mse_pred = mean_squared_error(y_test, pr1)
            mae_pred = mean_absolute_error(y_test, pr1)
            r2 = r2_score(y_test, pr1)


            plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="real difference", color='r', linewidth=1, s=1)
            plt.scatter( np.arange(0, len(y_test)),prediction, label="error Prediction ", color='turquoise', linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: TRAIN {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                'mae': mae_pred,
                'mse':mse_pred,
                'r2': r2,
                'prediccion': prediction
            })
            contador+=1

    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)

    matriz_mae = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="mae"
    )

    matriz_r2 = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="r2"
    )

    print("Matrix MAE:")
    print(matriz_mae)

    print("\nMatrix R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("SGD MAE(train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("SGD R2 (train vs test)")
    plt.show()

def treatment_SGDPARTIALseason(input,test,features,LABEL_SV):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL_SV].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL_SV].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL_SV].values

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index = df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road[LABEL_SV].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road[LABEL_SV].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]
    LABELS_ENTRENAMIENTO = [y_snow_road,  y_clear_road, y_clear_raining]
    INDICES = [y_test_snow_road_index,  y_test_clear_road_index, y_test_clear_raining_index ]
    DATOS_TEST = [X_TEST_snow_road,  X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road,  y_TEST_clear_road, y_TEST_clear_raining]

    results = []
    prediction = []
    condiciones = ["snow_road",  "clear_road", "clear_raining"]
    for est1, X_train, y_train, X_test, y_test in zip(
            condiciones,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training Conditions: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        lag = len(features)

        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        sgd2 = SGDRegressor(
            max_iter=2000,
            loss='epsilon_insensitive',  # Passive Aggressive Esto actúa como SVR online.
            learning_rate='pa1',
            eta0=1,
            alpha=0.001,
            shuffle=False,
            random_state=42,
            penalty=None
        )
        sgd2.fit(X_train, y_train)

        #coefs = np.abs(sgd2.coef_)  # shape (32,)
        # reshape → (n_features, n_lags)
        #coefs_reshaped = coefs.reshape(len(features), 2)
        #importance_per_feature = coefs_reshaped.sum(axis=1)

        contador=0
        for est2, X_test, y_test in zip(
                condiciones,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)  # create_lags
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Prediction Conditions: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            title_plt = f"Real {est1} vs Prediction {est2}" + " SGDRegressor with PartialFit"
            resultados_sgd_online, prediccion_sgd_online = evaluate_modelPH(title_plt, sgd2,
                                                                            X_train,
                                                                            y_train, X_test, y_test, 1, online=True)
            prediccion_sgd_onlinenversa = y_scaler.inverse_transform(
                np.array(prediccion_sgd_online).reshape(-1, 1)).flatten()
            prediction.append(prediccion_sgd_onlinenversa)

            y_test_inverse = y_scaler.inverse_transform(
                np.array(y_test).reshape(-1, 1)).flatten()
            mae_pred = mean_absolute_error(y_test_inverse, prediccion_sgd_onlinenversa)
            r2 = r2_score(y_test_inverse, prediccion_sgd_onlinenversa)

            '''df_sgd = pd.DataFrame({
                "feature": features,
                "importance": importance_per_feature
            }).sort_values(by="importance", ascending=False)

            values = df_sgd["importance"]
            colors = cm.viridis((values - values.min()) / (values.max() - values.min()))

            plt.figure(figsize=(8, 5))
            plt.barh(df_sgd["feature"], df_sgd["importance"],color=colors)
            plt.gca().invert_yaxis()
            plt.title(str(est1) + " Feature Importance - SGD PARTIAL")
            plt.show()'''

            '''plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(lag, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter(np.arange(lag, len(y_test)), prediccion_sgd_onlinenversa, label="Prediction SGD Partial", color='b', linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)'''

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                'mae': mae_pred,
                'r2': r2,
                'prediccion': prediccion_sgd_onlinenversa
            })
            contador+=1

    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)
    matriz_mae = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="mae"
    )

    matriz_r2 = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="r2"
    )

    print("Matrix MAE:")
    print(matriz_mae)

    print("\nMatrix R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("SGD PARTIAL FIT MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title(" SGD PARTIAL FIT R2 (train vs test)")
    plt.show()

def treatment_TKANseason(input,test,features,LABEL_SV):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL_SV].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL_SV].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL_SV].values

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index = df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road[LABEL_SV].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road[LABEL_SV].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]
    LABELS_ENTRENAMIENTO = [y_snow_road, y_clear_road, y_clear_raining]
    INDICES = [y_test_snow_road_index, y_test_clear_road_index, y_test_clear_raining_index]
    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]

    results = []
    prediction = []
    conditions = ["snow_road", "clear_road", "clear_raining"]
    for est1, X_train, y_train, X_test, y_test in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training Conditions: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        lag = len(features)
        XTRAIN_lag, yTRAIN_lag = create_sequences_flat(X_train, y_train, lags=lag)
        X_train, y_train = XTRAIN_lag, yTRAIN_lag

        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        contador=0
        for est2, X_test, y_test in zip(
                conditions,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_sequences_flat(X_test, y_test, lags=lag)
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Prediction Conditions: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            # T-KAN (aproximated with  MLP)
            tkan1 = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='identity',
                max_iter=8000,  #
                early_stopping=False,
                shuffle=False,
                n_iter_no_change=50,
                validation_fraction=0.0005,
                random_state=42
            )
            resultadosT_KAN, prediccionT_KAN = evaluate_modelPH("T-KAN (MLP)", tkan1, X_train, y_train, X_test, y_test,
                                                                2, online=False)
            prediccionT_KAN_inversa = y_scaler.inverse_transform(np.array(prediccionT_KAN).reshape(-1, 1)).flatten()
            prediction.append(prediccionT_KAN_inversa)

            y_test_inverse = y_scaler.inverse_transform(
                np.array(y_test).reshape(-1, 1)).flatten()
            mae_pred = mean_absolute_error(y_test_inverse, prediccionT_KAN_inversa)
            r2 = r2_score(y_test_inverse, prediccionT_KAN_inversa)

            '''plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter(np.arange(0, len(y_test)), prediccionT_KAN_inversa, label="Prediction TKAN", color='lightblue', linewidth=1, s=1)#
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)'''

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                'mae': mae_pred,
                'r2': r2,
                "T-KAN": prediccionT_KAN_inversa,
            })
            contador+=1

    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)
    matriz_mae = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="mae"
    )

    matriz_r2 = df_results.pivot(
        index="train_dataset",
        columns="test_dataset",
        values="r2"
    )

    print("Matrix MAE:")
    print(matriz_mae)

    print("\nMatrix R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("TKAN MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("TKAN R2 (train vs test)")
    plt.show()

def treatment_XGB_WHOLE(X,y,X_t,y_t,features):
    print("\n--- WHOLE DATASET ---")

    X_train = X[features].values
    y_train = y.values
    X_test = X_t[features].values
    y_test = y_t.values
    indice=y_t.index

    results = []
    prediction = []

    modelxgb = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    modelxgb.fit(X_train, y_train)

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    prediction_xgb=modelxgb.predict(X_test)
    prediction.append(prediction_xgb)

    mae_pred = mean_absolute_error(y_test, prediction_xgb)
    print(f"MAE : {mae_pred:.4f}")

    fechas_unicas = y_t.index.normalize().unique()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == fecha)

        horas = indice[mask]
        reales = y_test[mask]
        pred1 = prediction_xgb[mask]

        plt.figure(figsize=(12, 5))
        plt.title(f"Results for the day {fecha.date()}")
        plt.scatter(horas, reales, label="Real", color='r', linewidth=1, s=1)
        plt.scatter(horas, pred1, label="Prediction XGB", color='turquoise',
                    linewidth=1, s=1)
        plt.legend()
        plt.title(f"Time series: Real vs Prediction")
        plt.xlabel("Timestamp")
        plt.ylabel('SV')
        plt.show(block=True)

    results.append({
        'mae':mae_pred,
        'prediccion':prediction_xgb
    })


    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction

def treatment_SGD_WHOLE(X,y,X_t,y_t,features,label_target):
    print("\n--- WHOLE DATASET ---")

    X_train = X[features].values
    y_train = y.values
    X_test = X_t[features].values
    y_test = y_t.values
    indice = y_t.index

    results = []
    prediction = []

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    lag = len(features)

    XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)
    X_train, y_train = XTRAIN_lag, yTRAIN_lag
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)
    X_test, y_test = XTEST_lag, yTEST_lag
    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # SGD
    sgd1 = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',  # Passive Aggressive Esto actúa como SVR online.
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgd1.fit(X_train, y_train)

    resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH("SGDRegressor without PartialFit", sgd1,
                                                                      X_train, y_train, X_test, y_test, 0,
                                                                      online=False)
    prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
        np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
    prediction.append(prediccion_sgd_offlinenversa)
    mae_fit = resultados_sgd_offline['mae']
    r2fit = resultados_sgd_offline['r2']

    sgd2 = SGDRegressor(
        max_iter=2000,  # max_iter=1,
        # warm_start=True,
        loss='epsilon_insensitive',  # Passive Aggressive Esto actúa como SVR online.
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgd2.fit(X_train, y_train)
    resultados_sgd_online, prediccion_sgd_online = evaluate_modelPH("SGDRegressor with PartialFit", sgd2,
                                                                    X_train,
                                                                    y_train, X_test, y_test, 1, online=True)
    prediccion_sgd_onlinenversa = y_scaler.inverse_transform(
        np.array(prediccion_sgd_online).reshape(-1, 1)).flatten()
    prediction.append(prediccion_sgd_onlinenversa)

    mae_partial=resultados_sgd_online['mae']
    r2partial=resultados_sgd_online['r2']

    y_test = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
    fechas_unicas = y_t.index.normalize().unique()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice[lag:]).normalize() == fecha)
        maskdriftsgd=(pd.to_datetime(indice[lag:][resultados_sgd_offline['drifts']]).normalize() == fecha)
        maskdriftsgdpartial = (pd.to_datetime(indice[lag:][resultados_sgd_online['drifts']]).normalize() == fecha)

        horas = indice[lag:][mask]
        reales = y_test[mask]
        pred1 = prediccion_sgd_offlinenversa[mask]
        pred2 = prediccion_sgd_onlinenversa[mask]
        drift1 = indice[lag:][resultados_sgd_offline['drifts']][maskdriftsgd]
        drift2 = indice[lag:][resultados_sgd_online['drifts']][maskdriftsgdpartial]

        fig, ax = plt.subplots(figsize=(12, 5))
        plt.title(f"Results for the day {fecha.date()}")
        plt.scatter(horas, reales, label="Real", color='r', linewidth=1, s=1)
        plt.scatter(horas, pred1, label="Prediction SGD", color='turquoise',
                    linewidth=1, s=1)
        plt.scatter(horas, pred2, label="Prediction SGD Partial", color='b',
                    linewidth=1, s=1)

        cont=0
        for d in drift1:
            if cont == 0:
                plt.axvline(x=d, color='black', linestyle='--', alpha=0.7, label=' SGD Drift')
            else:
                plt.axvline(x=d, color='black', linestyle='--', alpha=0.7)
            cont += 1

        cont=0
        for d in drift2:
            if cont==0:
                plt.axvline(x=d, color='green', linestyle='--', alpha=0.7, label=' SGD Partial Drift')
            else:
                plt.axvline(x=d, color='green', linestyle='--', alpha=0.7)
            cont+=1

        #
        plt.text(
            1.05, 0.1,
            f"SGD MAE = {mae_fit:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.2,
            f"SGD R2 = {r2fit:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.3,
            f"SGD Partial MAE = {mae_partial:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.4,
            f"SGD Partial R2 = {r2partial:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )

        plt.tight_layout()

        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel(label_target)
        plt.show(block=True)

    results.append({
        "SGD": resultados_sgd_offline,
        "SGD PARtial": resultados_sgd_online
    })

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction,resultados_sgd_offline['drifts'],resultados_sgd_online['drifts']

def treatment_TKAN_WHOLE(X,y,X_t,y_t,features,label_target):
    print("\n--- WHOLE DATASET ---")
    X_train = X[features].values
    y_train = y.values
    X_test = X_t[features].values
    y_test = y_t.values
    indice = y_t.index

    results = []
    prediction = []

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    lag = len(features)

    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    XTRAIN_lag, yTRAIN_lag = create_sequences_flat(X_train, y_train, lags=lag)  # create_lags
    X_train, y_train = XTRAIN_lag, yTRAIN_lag

    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    XTEST_lag, yTEST_lag = create_sequences_flat(X_test, y_test, lags=lag)  # create_lags
    X_test, y_test = XTEST_lag, yTEST_lag

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # T-KAN (aproximated with  MLP)
    tkan1 = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='identity',
        max_iter=8000,  #
        early_stopping=False,
        shuffle=False,
        n_iter_no_change=50,
        validation_fraction=0.0005,
        random_state=42
    )
    tkan1.fit(X_train, y_train)
    resultadosT_KAN, prediccionT_KAN = evaluate_modelPH("T-KAN (MLP)", tkan1, X_train, y_train, X_test, y_test,
                                                        2, online=False)
    prediccionT_KAN_inversa = y_scaler.inverse_transform(np.array(prediccionT_KAN).reshape(-1, 1)).flatten()
    prediction.append(prediccionT_KAN_inversa)
    mae_tkan = resultadosT_KAN['mae']
    r2tkan = resultadosT_KAN['r2']

    y_test = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
    fechas_unicas = y_t.index.normalize().unique()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice[lag:]).normalize() == fecha)
        maskdriftsgd = (pd.to_datetime(indice[lag:][resultadosT_KAN['drifts']]).normalize() == fecha)

        horas = indice[lag:][mask]
        reales = y_test[mask]
        pred = prediccionT_KAN_inversa[mask]
        drift1 = indice[lag:][resultadosT_KAN['drifts']][maskdriftsgd]

        fig, ax = plt.subplots(figsize=(12, 5))
        plt.title(f"Results for the day {fecha.date()}")
        plt.scatter(horas, reales, label="Real", color='r', linewidth=1, s=1)
        plt.scatter(horas, pred, label="Prediction TKAN", color='lightblue',linewidth=1, s=1)
        '''
        cont = 0
        for d in drift1:
            if cont == 0:
                plt.axvline(x=d, color='brown', linestyle='--', alpha=0.7, label=' TKAN Drift')
            else:
                plt.axvline(x=d, color='brown', linestyle='--', alpha=0.7)
            cont += 1
        '''

        plt.text(
            1.05, 0.1,
            f"SGD MAE = {mae_tkan:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.2,
            f"SGD R2 = {r2tkan:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel(label_target)
        plt.show(block=True)

    results.append({
        "T-KAN": resultadosT_KAN,
    })

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction,resultadosT_KAN['drifts']

def train_model(df,features,target_label):
   X = df[features]
   y = df[target_label]
   model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42)
   model.fit(X, y)
   return model

def experiment_1CD(X,y,test,features,LABEL_SV):
    print("\n--- TRAIN with WHOLE DATASET:EXPERIMENT 1 ---")

    X_train = X[features].values
    y_train = y.values

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index = df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road[LABEL_SV].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road[LABEL_SV].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining[LABEL_SV].values

    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]
    INDICES = [y_test_snow_road_index, y_test_clear_road_index, y_test_clear_raining_index]

    results = []
    prediction = []
    conditions = ["TEST with snow road", "TEST with clear road", "TEST with raining"]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    lag = len(features)

    XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
    X_train, y_train = XTRAIN_lag, yTRAIN_lag
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    # SGD
    sgd1 = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )

    sgd1.fit(X_train, y_train)

    contador = 0
    for est2, X_test, y_test in zip(
            conditions,
            DATOS_TEST,
            LABELS_TEST
    ):
        XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)  # create_lags
        X_test, y_test = XTEST_lag, yTEST_lag
        X_test = x_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        print(f"{est2}")
        print("Train size:", X_train.shape)
        print("Test size:", X_test.shape)
        print(INDICES[contador].shape)

        title_plt = f" {est2}, for the whole trainng dataset."
        resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH(title_plt, sgd1,
                                                                          X_train, y_train, X_test, y_test, 0,
                                                                          online=False)
        prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
            np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
        prediction.append(prediccion_sgd_offlinenversa)

        y_test_inverse = y_scaler.inverse_transform(
            np.array(y_test).reshape(-1, 1)).flatten()
        mse_pred = mean_squared_error(y_test_inverse, prediccion_sgd_offlinenversa)
        mae_pred = mean_absolute_error(y_test_inverse, prediccion_sgd_offlinenversa)
        r2 = r2_score(y_test_inverse, prediccion_sgd_offlinenversa)

        results.append({
            "test_dataset": est2,
            'mae': mae_pred,
            'mse': mse_pred,
            'r2': r2,
        })

    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)
    print(df_results)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df_results,
        x="test_dataset",
        y="mae",
        color='b'
    )

    plt.title("Whole dataset for training. MAE Comparison by Conditions in the Test.")
    plt.xlabel("Test Condition")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.show()

def experiment_2CD(input,X_t,y_t,features,LABEL_SV):
    print("\n--- 3 models trained by different conditions, tested with WHOLE DATASET:EXPERIMENT 2 ---")

    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL_SV].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL_SV].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL_SV].values

    X_test = X_t[features].values
    y_test = y_t.values
    my_index = y_t.index

    lag = len(features)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)  # create_lags
    X_test, y_test = XTEST_lag, yTEST_lag
    X_test = x_scaler.fit_transform(X_test)
    y_test = y_scaler.fit_transform(y_test.reshape(-1, 1)).flatten()

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]  # , X_clear_raining
    LABELS_ENTRENAMIENTO = [y_snow_road, y_clear_road, y_clear_raining]  # , y_clear_raining

    results = []
    prediction = []
    coeficientes = {}
    conditions = ["Traing snow model", "Training clear model", "Training raining model"]  # , "clear_raining"
    for est1, X_train, y_train in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO
    ):
        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.transform(X_train)
        y_train = y_scaler.transform(y_train.reshape(-1, 1)).flatten()

        print(f"T{est1}")
        print("Train size:", X_train.shape)
        print("Test size:", X_test.shape)

        # SGD
        sgd1 = SGDRegressor(
            max_iter=2000,
            loss='epsilon_insensitive',
            learning_rate='pa1',
            eta0=1,
            alpha=0.001,
            shuffle=False,
            random_state=42,
            penalty=None
        )

        sgd1.fit(X_train, y_train)
        resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH(f"{est1}", sgd1,
                                                                          X_train, y_train, X_test, y_test, 0,
                                                                          online=False)
        prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
            np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
        prediction.append(prediccion_sgd_offlinenversa)
        mae_pred = resultados_sgd_offline['mae']
        r2= resultados_sgd_offline['r2']

        results.append({
            "train_dataset": est1,
            'mae': mae_pred,
            'r2': r2,
        })


    print("\n===== SUMMARY =====")
    df_results = pd.DataFrame(results)
    print(df_results)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df_results,
        x="train_dataset",
        y="mae",
        color='r'
    )

    plt.title("MAE Comparison with different training models by condition, and tested in the whole TEST dataset.")
    plt.xlabel("Training model")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.show()

def count_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return len(outliers), outliers

def VEL_THREE_models(input,X_t,y_t,features,LABEL,mod_cat,drifts):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL].values

    x_scalerSNOW = StandardScaler()
    y_scalerSNOW = StandardScaler()

    x_scalerCLEAR = StandardScaler()
    y_scalerCLEAR = StandardScaler()

    x_scalerRAIN = StandardScaler()
    y_scalerRAIN = StandardScaler()

    lag = len(features)

    XTRAIN_SNOW_lag, yTRAIN_SNOW_lag = create_lags_fast(X_snow_road, y_snow_road, lags=lag)
    X_train_SNOW, y_train_SNOW = XTRAIN_SNOW_lag, yTRAIN_SNOW_lag
    X_train_SNOW = x_scalerSNOW.fit_transform(X_train_SNOW)
    y_train_SNOW = y_scalerSNOW.fit_transform(y_train_SNOW.reshape(-1, 1)).flatten()

    sgdSNOW = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdSNOW.fit(X_train_SNOW, y_train_SNOW)#partial_fit

    XTRAIN_CLEAR_lag, yTRAIN_CLEAR_lag = create_lags_fast(X_clear_road, y_clear_road, lags=lag)
    X_train_CLEAR, y_train_CLEAR = XTRAIN_CLEAR_lag, yTRAIN_CLEAR_lag
    X_train_CLEAR = x_scalerCLEAR.fit_transform(X_train_CLEAR)
    y_train_CLEAR = y_scalerCLEAR.fit_transform(y_train_CLEAR.reshape(-1, 1)).flatten()

    sgdCLEAR = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdCLEAR.fit(X_train_CLEAR, y_train_CLEAR)#partial_fit

    XTRAIN_RAIN_lag, yTRAIN_RAIN_lag = create_lags_fast(X_clear_raining, y_clear_raining, lags=lag)
    X_train_RAIN, y_train_RAIN = XTRAIN_RAIN_lag, yTRAIN_RAIN_lag
    X_train_RAIN = x_scalerRAIN.fit_transform(X_train_RAIN)
    y_train_RAIN = y_scalerRAIN.fit_transform(y_train_RAIN.reshape(-1, 1)).flatten()

    sgdRAIN = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdRAIN.fit(X_train_RAIN, y_train_RAIN)#partial_fit

    X_test = X_t[features].values
    y_test = y_t.values
    my_index = y_t.index

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)
    X_test, y_test = XTEST_lag, yTEST_lag

    x_test_snow = x_scalerSNOW.transform(X_test)
    x_test_clear = x_scalerCLEAR.transform(X_test)
    x_test_rain = x_scalerRAIN.transform(X_test)

    y_test_snow = y_scalerSNOW.transform(y_test.reshape(-1, 1)).flatten()
    y_test_clear = y_scalerCLEAR.transform(y_test.reshape(-1, 1)).flatten()
    y_test_rain = y_scalerRAIN.transform(y_test.reshape(-1, 1)).flatten()

    prediction_WHOLE=[]
    segmentos = []
    start = 0

    for k, drift in enumerate(drifts):
        end = drift

        segmentos.append({
            "start": start,
            "end": end,
            "model": mod_cat[k]
        })

        start = drift

    segmentos.append({
        "start": start,
        "end": len(y_test),
        "model": mod_cat[-1]
    })

    results = []
    cont=0
    for seg in segmentos:
        start, end, model = seg["start"], seg["end"], seg["model"]

        if model == 1:
            scaler = y_scalerSNOW
            y_true_scaled = y_test_snow[start:end]
            prediction=sgdSNOW.predict(x_test_snow[start:end])
            y_pred_scaled = np.array(prediction)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
            y_pred = np.array([0 if (x < 0)|(x > 5) else x for x in y_pred])
            lim_max=144
        elif model == 2:
            scaler = y_scalerCLEAR
            y_true_scaled = y_test_clear[start:end]
            prediction = sgdCLEAR.predict(x_test_clear[start:end])
            y_pred_scaled = np.array(prediction)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
            y_pred = np.array([0 if x < 0 else x + 0.15 if 0 < x < 1.5 else x - 0.1 for x in y_pred])
            lim_max =107
        else:
            scaler = y_scalerRAIN
            y_true_scaled = y_test_rain[start:end]
            prediction = sgdRAIN.predict(x_test_rain[start:end])
            prediction = [abs(x) if x < 0 else x for x in prediction]
            y_pred_scaled = np.array(prediction)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
            lim_max =87
        if cont < 2:
            y_pred[-lag :] = 0

        prediction_WHOLE.append(y_pred)

        # errores
        errores = y_pred - y_true
        errores_abs = np.abs(errores)

        results.append(errores_abs)

        # stats
        print(f"\nSegmento {start}-{end} | Modelo: {model}")
        print("Bias:", np.mean(y_pred - y_true))

        media_error = np.mean(errores)
        mae = np.mean(errores_abs)
        max = np.max(errores_abs)

        p50 = np.percentile(errores_abs, 50)
        p90 = np.percentile(errores_abs, 90)
        p95 = np.percentile(errores_abs, 95)
        p99 = np.percentile(errores_abs, 99)

        num,out=count_outliers(errores_abs)
        print( num, "outliers")

        print("Mean error:", media_error)
        print("Maximum Error:", max)
        print("MAE:", mae)
        print("P50:", p50)
        print("P90:", p90)
        print("P95:", p95)
        print("P99:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"TARGET | Modelo: {model}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        plt.ylabel("Error")
        ax.set_ylim(0, lim_max)  # BOTH SAME SCALE
        plt.show(block=True)

        cont+=1

    return results,prediction_WHOLE

def VEL_THREE_modelserror(input,X_t,y_t,features,LABEL,mod_cat,drifts):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road =  10 *(df_snow_road[LABEL] - df_snow_road['lin_vel_odom_x']).values#10 *
    X_clear_road = df_clear_road[features].values
    y_clear_road =  10 *(df_clear_road[LABEL].values - df_clear_road['lin_vel_odom_x'])#10 *
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining =  10 *(df_clear_raining[LABEL].values - df_clear_raining['lin_vel_odom_x'])#10 *


    x_scalerSNOW = StandardScaler()
    y_scalerSNOW = StandardScaler()

    x_scalerCLEAR = StandardScaler()
    y_scalerCLEAR = StandardScaler()

    x_scalerRAIN = StandardScaler()
    y_scalerRAIN = StandardScaler()

    lag = len(features)

    XTRAIN_SNOW_lag, yTRAIN_SNOW_lag = create_lags_fast(X_snow_road, y_snow_road, lags=lag)
    X_train_SNOW, y_train_SNOW = XTRAIN_SNOW_lag, yTRAIN_SNOW_lag
    X_train_SNOW = x_scalerSNOW.fit_transform(X_train_SNOW)
    y_train_SNOW = y_scalerSNOW.fit_transform(y_train_SNOW.reshape(-1, 1)).flatten()

    sgdSNOW = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdSNOW.fit(X_train_SNOW, y_train_SNOW)#partial_fit

    XTRAIN_CLEAR_lag, yTRAIN_CLEAR_lag = create_lags_fast(X_clear_road, y_clear_road, lags=lag)
    X_train_CLEAR, y_train_CLEAR = XTRAIN_CLEAR_lag, yTRAIN_CLEAR_lag
    X_train_CLEAR = x_scalerCLEAR.fit_transform(X_train_CLEAR)
    y_train_CLEAR = y_scalerCLEAR.fit_transform(y_train_CLEAR.reshape(-1, 1)).flatten()

    sgdCLEAR = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdCLEAR.fit(X_train_CLEAR, y_train_CLEAR)#partial_fit

    XTRAIN_RAIN_lag, yTRAIN_RAIN_lag = create_lags_fast(X_clear_raining, y_clear_raining, lags=lag)
    X_train_RAIN, y_train_RAIN = XTRAIN_RAIN_lag, yTRAIN_RAIN_lag
    X_train_RAIN = x_scalerRAIN.fit_transform(X_train_RAIN)
    y_train_RAIN = y_scalerRAIN.fit_transform(y_train_RAIN.reshape(-1, 1)).flatten()

    sgdRAIN = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdRAIN.fit(X_train_RAIN, y_train_RAIN)#partial_fit

    X_test = X_t[features].values
    y_test =10 *(y_t.values-X_t['lin_vel_odom_x'])#10 *
    my_index = y_t.index

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)
    X_test, y_test = XTEST_lag, yTEST_lag

    x_test_snow = x_scalerSNOW.transform(X_test)
    x_test_clear = x_scalerCLEAR.transform(X_test)
    x_test_rain = x_scalerRAIN.transform(X_test)

    y_test_snow = y_scalerSNOW.transform(y_test.reshape(-1, 1)).flatten()
    y_test_clear = y_scalerCLEAR.transform(y_test.reshape(-1, 1)).flatten()
    y_test_rain = y_scalerRAIN.transform(y_test.reshape(-1, 1)).flatten()

    prediction_WHOLE=[]
    segmentos = []
    start = 0

    for k, drift in enumerate(drifts):
        end = drift

        segmentos.append({
            "start": start,
            "end": end,
            "model": mod_cat[k]
        })

        start = drift

    segmentos.append({
        "start": start,
        "end": len(y_test),
        "model": mod_cat[-1]
    })

    results = []
    cont=0
    for seg in segmentos:
        start, end, model = seg["start"], seg["end"], seg["model"]

        if model == 1:
            scaler = y_scalerSNOW
            y_true_scaled = y_test_snow[start:end]
            prediction=sgdSNOW.predict(x_test_snow[start:end])
            y_pred_scaled = np.array(prediction)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
            y_pred = np.array([0 if (x < 0)|(x > 5) else x for x in y_pred])
            lim_max=144
        elif model == 2:
            scaler = y_scalerCLEAR
            y_true_scaled = y_test_clear[start:end]
            prediction = sgdCLEAR.predict(x_test_clear[start:end])
            y_pred_scaled = np.array(prediction)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
            y_pred = np.array([0 if x < 0 else x + 0.15 if 0 < x < 1.5 else x - 0.1 for x in y_pred])
            lim_max =107
        else:
            scaler = y_scalerRAIN
            y_true_scaled = y_test_rain[start:end]
            prediction = sgdRAIN.predict(x_test_rain[start:end])
            prediction = [abs(x) if x < 0 else x for x in prediction]
            y_pred_scaled = np.array(prediction)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
            lim_max =87
        if cont < 2:
            y_pred[-lag :] = 0

        prediction_WHOLE.append(y_pred)

        # errores
        errores = y_pred - y_true
        errores_abs = np.abs(errores)

        results.append(errores_abs)

        # stats
        print(f"\nSegmento {start}-{end} | Modelo: {model}")
        print("Bias:", np.mean(y_pred - y_true))

        media_error = np.mean(errores)
        mae = np.mean(errores_abs)
        max = np.max(errores_abs)

        p50 = np.percentile(errores_abs, 50)
        p90 = np.percentile(errores_abs, 90)
        p95 = np.percentile(errores_abs, 95)
        p99 = np.percentile(errores_abs, 99)

        num,out=count_outliers(errores_abs)
        print( num, "outliers")

        print("Mean error:", media_error)
        print("Maximum Error:", max)
        print("MAE:", mae)
        print("P50:", p50)
        print("P90:", p90)
        print("P95:", p95)
        print("P99:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"TARGET | Modelo: {model}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        plt.ylabel("Error")
        ax.set_ylim(0, lim_max)  # BOTH SAME SCALE
        plt.show(block=True)

        cont+=1

    return results,prediction_WHOLE

def VEL_THREE_modelselastic(input,X_t,y_t,features,LABEL,mod_cat,drifts):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = 10 *(df_snow_road[LABEL]-df_snow_road['lin_vel_odom_x']).values#10 *
    X_clear_road = df_clear_road[features].values
    y_clear_road = 10 *(df_clear_road[LABEL]-df_clear_road['lin_vel_odom_x']).values#10 *
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = 10 *(df_clear_raining[LABEL]-df_clear_raining['lin_vel_odom_x']).values#10 *

    elasticSNOW = ElasticNet(alpha=0.1,l1_ratio=0.2,max_iter=10000,
                tol=1e-5, random_state=42,selection='cyclic' )
    elasticSNOW.fit(X_snow_road, y_snow_road)

    elasticCLEAR = ElasticNet(alpha=0.1,l1_ratio=0.2,max_iter=10000,
                tol=1e-5, random_state=42,selection='cyclic' )
    elasticCLEAR.fit(X_clear_road, y_clear_road)

    elasticRAIN = ElasticNet(alpha=0.1,l1_ratio=0.2,max_iter=10000,
                tol=1e-5, random_state=42,selection='cyclic' )
    elasticRAIN.fit(X_clear_raining, y_clear_raining)

    X_test = X_t[features].values
    y_test = 10 *(y_t-X_t['lin_vel_odom_x']).values#10 *
    y_test=np.clip(y_test, -30, 30)

    prediction_WHOLE=[]
    segmentos = []
    start = 0

    for k, drift in enumerate(drifts):
        end = drift

        segmentos.append({
            "start": start,
            "end": end,
            "model": mod_cat[k]
        })

        start = drift

    segmentos.append({
        "start": start,
        "end": len(y_test),
        "model": mod_cat[-1]
    })

    results = []
    cont=0
    for seg in segmentos:
        start, end, model = seg["start"], seg["end"], seg["model"]#LIN_VEL_MAX=[91,144,112]

        if model == 1:
            prediction=elasticSNOW.predict(X_test[start:end])
            lim_max=144
        elif model == 2:
            prediction = elasticCLEAR.predict(X_test[start:end])
            lim_max =112
        else:
            prediction = elasticRAIN.predict(X_test[start:end])
            lim_max =91

        prediction_WHOLE.append(prediction)

        # errores
        errores = prediction - y_test[start:end]
        errores_abs = np.abs(prediction - y_test[start:end])
        results.append(errores_abs)

        i = np.argmax(np.abs(errores))
        i= 3867

        '''plt.plot(y_t[start:end],errores,color='r',label='error')
        plt.plot(y_t[start:end].index,X_t['lin_vel_odom_x'][start:end].values,color='g',label='odom')
        plt.plot(y_t[start:end].index,y_t[start:end].values,color='b',label='GT')
        plt.plot(y_t[start:end].index,prediction,color='orange',label='pred')
        plt.legend()
        plt.show(block=True)'''

        # stats
        print(f"\nSegmento {start}-{end} | Modelo: {model}")

        media_error = np.mean(errores)
        mae = np.mean(errores_abs)
        max = np.max(errores_abs)

        p50 = np.percentile(errores_abs, 50)
        p90 = np.percentile(errores_abs, 90)
        p95 = np.percentile(errores_abs, 95)
        p99 = np.percentile(errores_abs, 99)

        num,out=count_outliers(errores_abs)
        print( num, "outliers")

        print("Mean error:", media_error)
        print("Maximum Error:", max)
        print("MAE:", mae)
        print("P50:", p50)
        print("P90:", p90)
        print("P95:", p95)
        print("P99:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"TARGET | Modelo: {model}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        plt.ylabel("Error")
        ax.set_ylim(0, lim_max)  # BOTH SAME SCALE
        plt.show(block=True)

        cont+=1

    return results,prediction_WHOLE

def THREE_models(input,X_t,y_t,features,LABEL_SV):
    print("\n--- Cross-season generalization ---")
    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road[LABEL_SV].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road[LABEL_SV].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining[LABEL_SV].values

    x_scalerSNOW = StandardScaler()
    y_scalerSNOW = StandardScaler()

    x_scalerCLEAR = StandardScaler()
    y_scalerCLEAR = StandardScaler()

    x_scalerRAIN = StandardScaler()
    y_scalerRAIN = StandardScaler()

    lag = len(features)

    XTRAIN_SNOW_lag, yTRAIN_SNOW_lag = create_lags_fast(X_snow_road, y_snow_road, lags=lag)
    X_train_SNOW, y_train_SNOW = XTRAIN_SNOW_lag, yTRAIN_SNOW_lag
    X_train_SNOW = x_scalerSNOW.fit_transform(X_train_SNOW)
    y_train_SNOW = y_scalerSNOW.fit_transform(y_train_SNOW.reshape(-1, 1)).flatten()

    sgdSNOW = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdSNOW.fit(X_train_SNOW, y_train_SNOW)#partial_fit

    XTRAIN_CLEAR_lag, yTRAIN_CLEAR_lag = create_lags_fast(X_clear_road, y_clear_road, lags=lag)
    X_train_CLEAR, y_train_CLEAR = XTRAIN_CLEAR_lag, yTRAIN_CLEAR_lag
    X_train_CLEAR = x_scalerCLEAR.fit_transform(X_train_CLEAR)
    y_train_CLEAR = y_scalerCLEAR.fit_transform(y_train_CLEAR.reshape(-1, 1)).flatten()

    sgdCLEAR = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdCLEAR.fit(X_train_CLEAR, y_train_CLEAR)#partial_fit

    XTRAIN_RAIN_lag, yTRAIN_RAIN_lag = create_lags_fast(X_clear_raining, y_clear_raining, lags=lag)
    X_train_RAIN, y_train_RAIN = XTRAIN_RAIN_lag, yTRAIN_RAIN_lag
    X_train_RAIN = x_scalerRAIN.fit_transform(X_train_RAIN)
    y_train_RAIN = y_scalerRAIN.fit_transform(y_train_RAIN.reshape(-1, 1)).flatten()

    sgdRAIN = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgdRAIN.fit(X_train_RAIN, y_train_RAIN)#partial_fit

    X_test = X_t[features].values
    y_test = y_t.values
    indice = y_t.index

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)
    X_test, y_test = XTEST_lag, yTEST_lag

    x_test_snow = x_scalerSNOW.transform(X_test)
    x_test_clear = x_scalerCLEAR.transform(X_test)
    x_test_rain = x_scalerRAIN.transform(X_test)

    y_test_snow = y_scalerSNOW.transform(y_test.reshape(-1, 1)).flatten()
    y_test_clear = y_scalerCLEAR.transform(y_test.reshape(-1, 1)).flatten()
    y_test_rain = y_scalerRAIN.transform(y_test.reshape(-1, 1)).flatten()

    X_t_copy=X_t.copy()
    X_t_copy.index = pd.to_datetime(X_t_copy.index)
    df_first =X_t_copy.loc[X_t_copy.groupby(X_t_copy.index.date).head(1).index]
    print(X_t_copy.index.get_loc("2025-11-03 23:37:24.368647936"),'2025-11-03 23:37:24.368647936')
    print(X_t_copy.index.get_loc("2024-11-28 14:44:26.811245056"),'2024-11-28 14:44:26.811245056')
    print(X_t_copy.index.get_loc("2025-08-20 14:42:36.519282176"),'2025-08-20 14:42:36.519282176')

    results = []
    prediction = []
    prediction_WHOLE = []
    drifts = []
    drifts_date = []
    mod_category=[]
    model_used = []
    prev_model = None
    stability_threshold = len(features)+1
    model_buffer = []

    cont=0
    tmp=0
    for i in range(len(X_test)):
        xsnow = x_test_snow[i].reshape(1, -1)
        xclear = x_test_clear[i].reshape(1, -1)
        xrain = x_test_rain[i].reshape(1, -1)

        y_predsnow = float((sgdSNOW.predict(xsnow))[0])
        y_predclear = float((sgdCLEAR.predict(xclear))[0])
        y_predrain = float((sgdRAIN.predict(xrain))[0])

        if (y_predsnow > 30):
            pred = y_predclear
            current_model = "CLEAR"
        elif ((y_predclear < -20) & (y_predrain > 25)):
            pred = y_predsnow
            current_model = "SNOW"
        else:
            pred = y_predrain
            current_model = "RAIN"

        '''if((y_predsnow <-10) & (y_predrain > 100)):
            pred=y_predclear
            current_model = "CLEAR"
        elif ((y_predclear>20)&(y_predrain>35)):
            pred=y_predsnow
            current_model = "SNOW"
        else:
            pred=y_predrain
            current_model = "RAIN"'''

        prediction.append(pred)
        prediction_WHOLE.append(pred)
        model_used.append(current_model)
        model_buffer.append(current_model)

        if len(model_buffer) > stability_threshold:
            model_buffer.pop(0)

        if len(set(model_buffer)) == 1:
            stable_model = model_buffer[0]

            if prev_model is not None and stable_model != prev_model:
                print(f"Cambio real en índice {i}")
                print(f"CORRESPONDENCIA {indice[i]}")
                print(prev_model,stable_model)
                drifts.append(i)
                drifts_date.append(indice[i])
                cat=0
                if cont==0:
                    if prev_model=='RAIN':
                        prediccion_inenversa = y_scalerRAIN.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
                        y_testtmp = y_scalerRAIN.inverse_transform(np.array(y_test_rain[:i + 1]).reshape(-1, 1)).flatten()
                        cat=3
                        lim_max = 44
                    elif prev_model == 'CLEAR':
                        prediccion_inenversa = y_scalerCLEAR.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
                        y_testtmp = y_scalerCLEAR.inverse_transform(np.array(y_test_clear[:i + 1]).reshape(-1, 1)).flatten()
                        cat=2
                        lim_max = 152
                    else:
                        prediccion_inenversa = y_scalerSNOW.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
                        y_testtmp = y_scalerSNOW.inverse_transform(np.array(y_test_snow[:i + 1]).reshape(-1, 1)).flatten()
                        cat=1
                        lim_max = 161
                    mod_category.append(cat)
                    prediccion_inenversa = np.clip(prediccion_inenversa, -100, 100)
                    bias = np.mean(prediccion_inenversa - y_testtmp)
                    print("Bias:", bias)
                elif cont==1:
                    if prev_model=='RAIN':
                        prediccion_inenversa = y_scalerRAIN.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
                        y_testtmp = y_scalerRAIN.inverse_transform(np.array(y_test_rain[tmp+1:i + 1]).reshape(-1, 1)).flatten()
                        cat=3
                        lim_max = 44
                    elif prev_model == 'CLEAR':
                        prediccion_inenversa = y_scalerCLEAR.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
                        y_testtmp = y_scalerCLEAR.inverse_transform(np.array(y_test_clear[tmp+1:i + 1]).reshape(-1, 1)).flatten()
                        cat = 2
                        lim_max = 152
                    else:
                        prediccion_inenversa = y_scalerSNOW.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
                        y_testtmp = y_scalerSNOW.inverse_transform(np.array(y_test_snow[tmp+1:i + 1]).reshape(-1, 1)).flatten()
                        cat = 1
                        lim_max = 161

                    mod_category.append(cat)
                    prediccion_inenversa = np.clip(prediccion_inenversa, -100, 100)
                    bias = np.mean(prediccion_inenversa - y_testtmp)
                    print("Bias:", bias)

                errores = prediccion_inenversa - y_testtmp
                errores_abs = np.abs(errores)

                results.append(errores)

                media_error = np.mean(errores)
                mae = np.mean(errores_abs)
                max = np.max(errores_abs)

                num, out = count_outliers(errores_abs)
                print(num, "outliers")

                p50 = np.percentile(errores_abs, 50)
                p90 = np.percentile(errores_abs, 90)
                p95 = np.percentile(errores_abs, 95)
                p99 = np.percentile(errores_abs, 99)

                print("Mean error:", media_error)
                print("Maximum Error:", max)
                print("MAE:", mae)
                print("P50:", p50)
                print("P90:", p90)
                print("P95:", p95)
                print("P99:", p99)

                fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
                fig.suptitle('Absolute Error Distribution Boxplot', fontsize=16)
                plt.title(f"{LABEL_SV}   Results until the drift is detected.")
                plt.boxplot(errores_abs)
                #plt.boxplot(errores_abs, showfliers=False)
                plt.ylabel("Error")
                ax.set_ylim(0, lim_max)  # BOTH SAME SCALE
                plt.show(block=True)

                prediction = []
                tmp=i
                cont+=1

            prev_model = stable_model

    if current_model == 'RAIN':
        prediccion_inenversa = y_scalerRAIN.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
        y_testtmp = y_scalerRAIN.inverse_transform(np.array(y_test_rain[tmp + 1:]).reshape(-1, 1)).flatten()
        cat=3
        lim_max = 44
    elif current_model == 'CLEAR':
        prediccion_inenversa = y_scalerCLEAR.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
        y_testtmp = y_scalerCLEAR.inverse_transform(np.array(y_test_clear[tmp + 1:]).reshape(-1, 1)).flatten()
        cat=2
        lim_max = 152
    else:
        prediccion_inenversa = y_scalerSNOW.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
        y_testtmp = y_scalerSNOW.inverse_transform(np.array(y_test_snow[tmp + 1:]).reshape(-1, 1)).flatten()
        cat=1
        lim_max = 161

    mod_category.append(cat)
    prediccion_inenversa = np.clip(prediccion_inenversa, -100, 100)
    bias = np.mean(prediccion_inenversa - y_testtmp)
    print("Bias:", bias)

    mae = mean_absolute_error(y_testtmp, prediccion_inenversa)
    print(mae)
    errores = prediccion_inenversa - y_testtmp
    errores_abs = np.abs(errores)

    results.append(errores)

    media_error = np.mean(errores)
    mae = np.mean(errores_abs)
    max = np.max(errores_abs)

    num, out = count_outliers(errores_abs)
    print(num, "outliers")

    p50 = np.percentile(errores_abs, 50)
    p90 = np.percentile(errores_abs, 90)
    p95 = np.percentile(errores_abs, 95)
    p99 = np.percentile(errores_abs, 99)

    print("Mean error:", media_error)
    print("Maximum Error:", max)
    print("MAE:", mae)
    print("P50:", p50)
    print("P90:", p90)
    print("P95:", p95)
    print("P99:", p99)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
    fig.suptitle('Absolute Error Distribution Boxplot', fontsize=16)
    plt.title(f"{LABEL_SV}   Results until the drift is detected.")
    plt.boxplot(errores_abs)
    #plt.boxplot(errores_abs, showfliers=False)
    ax.set_ylim(0, lim_max)  # BOTH SAME SCALE
    plt.ylabel("Error")
    plt.show()

    return results,prediction_WHOLE,mod_category,drifts,drifts_date

def ONE_model(X,y,X_t,y_t,features,LABEL_SV):
    X_train = X[features].values
    y_train = y.values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    lag = len(features)

    XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)
    X_train, y_train = XTRAIN_lag, yTRAIN_lag
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    # SGD
    sgd = SGDRegressor(
        max_iter=2000,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        shuffle=False,
        random_state=42,
        penalty=None
    )
    sgd.fit(X_train, y_train)

    X_test = X_t[features].values
    y_test = y_t.values
    indice = y_t.index

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)
    X_test, y_test = XTEST_lag, yTEST_lag
    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    prediction=sgd.predict(X_test)

    prediccion_inenversa = y_scaler.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
    y_test = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()

    fechas_unicas = y_t.index.normalize().unique()
    results=[]
    cont=0
    if LABEL_SV=='SV':
        max_lim=SV_MAX
    else:
        max_lim=LIN_VEL_MAX
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice[lag:]).normalize() == fecha)
        horas = indice[lag:][mask]
        reales = y_test[mask]
        pred1 = prediccion_inenversa[mask]

        errores = reales - pred1
        print("Bias:", np.mean(reales - pred1))
        errores_abs = np.abs(errores)

        results.append(errores)

        media_error = np.mean(errores)
        mae = np.mean(errores_abs)
        max=np.max(errores_abs)

        num, out = count_outliers(errores_abs)
        print(num, "outliers")

        p50 = np.percentile(errores_abs, 50)
        p90 = np.percentile(errores_abs, 90)
        p95 = np.percentile(errores_abs, 95)
        p99 = np.percentile(errores_abs, 99)

        print("Mean error:", media_error)
        print("Maximum Error:", max)
        print("MAE:", mae)
        print("P50:", p50)
        print("P90:", p90)
        print("P95:", p95)
        print("P99:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Errors Distribution Boxplot', fontsize=16)
        plt.title(f"{LABEL_SV}   Results for the day {fecha.date()}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        ax.set_ylim(0, max_lim[cont])  # BOTH SAME SCALE
        plt.ylabel("Error")
        plt.show(block=True)
        cont+=1

    return results,prediccion_inenversa

def ONE_modelelastic(X,y,X_t,y_t,features,LABEL_SV):
    X_train = X[features].values
    y_train = 10 *( y.values - X['lin_vel_odom_x'])#10 *

    #elastic = ElasticNet(alpha=0.01,max_iter=10000, l1_ratio=0.3,selection='random', random_state=42)
    elastic =ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=10000,
               tol=1e-5, random_state=42, selection='cyclic')
    elastic.fit(X_train, y_train)

    X_test = X_t[features].values
    y_test = 10 *(y_t.values - X_t['lin_vel_odom_x'])#10 *
    y_test=np.clip(y_test, -30, 30)
    indice = y_t.index

    prediction=elastic.predict(X_test)

    fechas_unicas = y_t.index.normalize().unique()
    results=[]
    cont=0
    max_lim=[91,144,112]
    pred_total = prediction.copy()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == fecha)
        horas = indice[mask]
        reales = y_test[mask]
        prederror = pred_total[mask]
        odom_lin_vel = X_t[mask]['lin_vel_odom_x']
        inferencia = odom_lin_vel + prederror /10

        errores = reales - inferencia
        print("Bias:", np.mean(reales - prederror))
        errores_abs = np.abs(errores)

        results.append(errores)

        media_error = np.mean(errores)
        mae = np.mean(errores_abs)
        max=np.max(errores_abs)

        num, out = count_outliers(errores_abs)
        print(num, "outliers")

        p50 = np.percentile(errores_abs, 50)
        p90 = np.percentile(errores_abs, 90)
        p95 = np.percentile(errores_abs, 95)
        p99 = np.percentile(errores_abs, 99)

        print("Mean error:", media_error)
        print("Maximum Error:", max)
        print("MAE:", mae)
        print("P50:", p50)
        print("P90:", p90)
        print("P95:", p95)
        print("P99:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"{LABEL_SV}   Results for the day {fecha.date()}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        ax.set_ylim(0, max_lim[cont])  # BOTH SAME SCALE
        plt.ylabel("Error")
        plt.show(block=True)

        '''fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str('ONE MODEL' + str(fecha))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"LINEAR VELOCITY {fecha.date()}")
        plt.plot(reales, label="Ground Truth", color='r', linewidth=1)
        plt.plot(inferencia, label="Prediction", color='blue', linewidth=1)
        plt.plot(odom_lin_vel, color='turquoise', label='Odometry')
        plt.legend()
        plt.show(block=True)'''
        cont+=1

    return results,prediction

def model_Slip_index_OLD(input,test,features):
    print("\n--- Three slip index calculating ---")

    df_snow_road = input[1]
    df_clear_road = input[2]
    df_clear_raining = input[3]

    X_snow_road = df_snow_road[features].values
    y_snow_road = df_snow_road['SV'].values
    X_clear_road = df_clear_road[features].values
    y_clear_road = df_clear_road['SV'].values
    X_clear_raining = df_clear_raining[features].values
    y_clear_raining = df_clear_raining['SV'].values

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    X_TEST_snow_road = df_TEST_snow_road[features].values
    y_test_snow_road_index = df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road['SV'].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road['SV'].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining['SV'].values

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]
    LABELS_ENTRENAMIENTO = [y_snow_road, y_clear_road, y_clear_raining]
    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]

    predictionsnow = []
    predictionrain = []
    predictionclear = []

    trainsnow = []
    trainrain = []
    trainclear = []

    conditions = ["snow_road", "clear_road", "raining"]
    for est1, X_train, y_train, X_test, y_test in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training Conditions: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        lag = len(features)

        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        sgd = SGDRegressor(
            max_iter=2000,
            loss='epsilon_insensitive',
            learning_rate='pa1',
            eta0=1,
            alpha=0.001,
            shuffle=False,
            random_state=42,
            penalty=None
        )

        sgd.fit(X_train, y_train)

        cont2=0
        for est2, X_test, y_test in zip(
                conditions,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)  # create_lags
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Prediction Conditions: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)

            pr1 = sgd.predict(X_test)
            pr1inenversa = y_scaler.inverse_transform(np.array(pr1).reshape(-1, 1)).flatten()
            pr1tr = sgd.predict(X_train)
            pr1_TR_inenversa = y_scaler.inverse_transform(np.array(pr1tr).reshape(-1, 1)).flatten()

            if cont2 == 0:
                trainsnow.append(pr1_TR_inenversa)
                predictionsnow.append(pr1inenversa)
            elif cont2 == 1:
                predictionclear.append(pr1inenversa)
                trainclear.append(pr1_TR_inenversa)
            else:
                predictionrain.append(pr1inenversa)
                trainrain.append(pr1_TR_inenversa)

            y_test_inverse = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
            y_train_inverse = y_scaler.inverse_transform(np.array(y_train).reshape(-1, 1)).flatten()

            mae_TR_pred = mean_absolute_error(y_train_inverse, pr1_TR_inenversa)
            mae_pred = mean_absolute_error(y_test_inverse, pr1inenversa)
            print(mae_pred,mae_TR_pred)
            cont2 += 1
            '''plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter(np.arange(0, len(y_test)), pr1, label="Prediction SGD",
                        color='turquoise', linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)'''

    return  predictionsnow,predictionclear,predictionrain,trainsnow,trainclear,trainrain

def predict_dataset(df_dict,features,lag,models,scalers, dataset_name):
    results = {}
    conditions = ["snow_road", "clear_road", "raining"]
    for cond_data in conditions:
        df = df_dict[cond_data]
        X = df[features].values
        y = df["SV"].values

        X_lag, y_lag = create_lags_fast(X, y, lags=lag)
        df_result = pd.DataFrame(index=df.index[lag:])

        for cond_model in conditions:
            model = models[cond_model]
            x_scaler, y_scaler = scalers[cond_model]

            X_scaled = x_scaler.transform(X_lag)

            pred = model.predict(X_scaled)
            pred_inv = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

            df_result[f"SV_{cond_model}_pred"] = pred_inv

        df_result["SV_real"] = y_lag
        results[cond_data] = df_result

        for cond_model in conditions:
            mae = mean_absolute_error(
                df_result["SV_real"],
                df_result[f"SV_{cond_model}_pred"]
            )
            print(f"{dataset_name} | Data={cond_data} | Model={cond_model} | MAE={mae:.4f}")

    return results

def train_models(input,features):
    train_dfs = {
        "snow_road": input[1],
        "clear_road": input[2],
        "raining": input[3]
    }

    models = {}
    scalers = {}

    lag = len(features)

    conditions = ["snow_road", "clear_road", "raining"]
    for cond in conditions:
        df_train = train_dfs[cond]

        X_train = df_train[features].values
        y_train = df_train["SV"].values

        # LAGS
        X_lag, y_lag = create_lags_fast(X_train, y_train, lags=lag)

        # ESCALADO
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = x_scaler.fit_transform(X_lag)
        y_scaled = y_scaler.fit_transform(y_lag.reshape(-1, 1)).flatten()

        # MODELO
        model = SGDRegressor(
            max_iter=2000,
            loss="epsilon_insensitive",
            learning_rate="pa1",
            eta0=1,
            alpha=0.001,
            shuffle=False,
            random_state=42,
            penalty=None
        )

        model.fit(X_scaled, y_scaled)

        # GUARDAR
        models[cond] = model
        scalers[cond] = (x_scaler, y_scaler)
    return models, scalers

def model_Slip_index(input,test,features):
    print("\n--- Three slip index calculating ---")
    my_models,myscalers=train_models(input,features)
    train_dfs = {
        "snow_road": input[1],
        "clear_road": input[2],
        "raining": input[3]
    }

    test_dfs = {
        "snow_road": test[1],
        "clear_road": test[2],
        "raining": test[3]
    }

    lag = len(features)
    train_results = predict_dataset(train_dfs,features,lag,my_models,myscalers, "TRAIN")
    test_results = predict_dataset(test_dfs,features,lag,my_models,myscalers ,"TEST")

    return  train_results,test_results

def conditional_model(X,y,X_t,y_t,features,LABEL_SV):
    X_train = X[features].values
    y_train = 10 *( y.values - X['lin_vel_odom_x'])

    #elastic = ElasticNet(alpha=0.01,max_iter=10000, l1_ratio=0.3,selection='random', random_state=42)
    elastic =ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=10000,
               tol=1e-5, random_state=42, selection='cyclic')
    elastic.fit(X_train, y_train)

    X_test = X_t[features].values
    y_test = 10 *(y_t.values - X_t['lin_vel_odom_x'])
    y_test=np.clip(y_test, -30, 30)
    indice = y_t.index

    prediction=elastic.predict(X_test)

    y_t.index = pd.to_datetime(y_t.index)
    fechas_unicas = y_t.index.normalize().unique()
    results=[]
    cont=0
    max_lim=[91,144,112]#
    pred_total = prediction.copy()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == fecha)
        reales = y_test[mask]
        prederror = pred_total[mask]
        odom_lin_vel = X_t[mask]['lin_vel_odom_x']
        inferencia = odom_lin_vel + prederror /10

        errores = reales - inferencia
        print("Bias:", np.mean(reales - prederror))
        errores_abs = np.abs(errores)

        results.append(errores)

        media_error = np.mean(errores)
        mae = np.mean(errores_abs)
        max=np.max(errores_abs)

        num, out = count_outliers(errores_abs)
        print(num, "outliers")

        p50 = np.percentile(errores_abs, 50)
        p90 = np.percentile(errores_abs, 90)
        p95 = np.percentile(errores_abs, 95)
        p99 = np.percentile(errores_abs, 99)

        print("Mean error:", media_error)
        print("Maximum Error:", max)
        print("MAE:", mae)
        print("P50:", p50)
        print("P90:", p90)
        print("P95:", p95)
        print("P99:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"{LABEL_SV}   Results for the day {fecha.date()}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        ax.set_ylim(0, max_lim[cont])  # BOTH SAME SCALE
        plt.ylabel("Error")
        plt.show(block=True)

        '''fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str('ONE MODEL' + str(fecha))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"LINEAR VELOCITY {fecha.date()}")
        plt.plot(reales, label="Ground Truth", color='r', linewidth=1)
        plt.plot(inferencia, label="Prediction", color='blue', linewidth=1)
        plt.plot(odom_lin_vel, color='turquoise', label='Odometry')
        plt.legend()
        plt.show(block=True)'''
        cont+=1

    return results,prediction

def merge_predictions(original_dfs,lag, results_dict):
    merged_list = []
    conditions = [1,2,3]
    for cond in conditions:
        df_original = original_dfs[cond]
        df_pred = results_dict[cond]

        df_original_cut = df_original.iloc[lag:].copy()
        df_original_cut = df_original_cut.loc[df_pred.index]

        for col in df_pred.columns:
            df_original_cut[col] = df_pred[col]

        #df_original_cut = df_original_cut.drop(["SV", "Soil_type"], axis=1)
        merged_list.append(df_original_cut)

    final_df = pd.concat(merged_list).sort_index()
    return final_df

def IMPROVEMENT_VISUALIZATION(X_t,y_t,pred,features,lab_target,title):
    indice = y_t.index

    lag = len(features)

    fechas_unicas = y_t.index.normalize().unique()
    pred_total=pred.copy()
    for mydate in fechas_unicas:
        mask = (pd.to_datetime(indice[lag:]).normalize() == mydate)
        predt = pd.Series(pred_total, index=indice[lag:])#PREDICTION OF LINEAR VELOCITY

        horas = indice[lag:][mask]
        reales = y_t[lag:][mask]
        odom_lin_vel = X_t[lag:][mask]['lin_vel_odom_x']
        odom_ang_vel = X_t[lag:][mask]['ang_vel_odom_z']
        imu_ang_vel = X_t[lag:][mask]['ang_vel_imu_z']
        predt = predt[mask]

        error_ODOM = odom_lin_vel - reales
        error_PRED = predt - reales

        if((mydate!=pd.Timestamp('2025-11-03 00:00:00')) & (mydate!=pd.Timestamp('2024-11-28 00:00:00'))):
            color='red'
        elif(mydate!=pd.Timestamp('2025-11-03 00:00:00')):
            color = 'yellow'
        else:
            color = 'blue'

        fecha_str = mydate.date().strftime("%Y-%m-%d")
        path_odom = NEW_FOMO_PATH + f"{fecha_str}_{color}_odom_position.csv"
        path_index = NEW_FOMO_PATH + f"{fecha_str}_{color}_gt_raw_index.csv"
        error_file = NEW_FOMO_PATH + f"{fecha_str}_{color}_error.csv"
        trajectoryGT = file_interface.read_tum_trajectory_file(NEW_FOMO_PATH + f"{fecha_str}_{color}_trajectory.txt")
        trajectoryGT.reduce_to_ids(list(range(lag+1, trajectoryGT.num_poses)))

        pos_odom = np.loadtxt(path_odom, delimiter=",")
        pos_gt = trajectoryGT.positions_xyz
        pos_gt = pos_gt[~np.isnan(pos_gt).any(axis=1)]
        index_gt=np.loadtxt(path_index, delimiter=",")

        print(odom_ang_vel.shape[0], reales.shape[0], odom_lin_vel.shape[0], predt.shape[0])

        delta_ts = horas.diff().values.astype(np.int64) / 1e9
        print(delta_ts.shape[0])

        p_rel_gt_rec, p_gt_rec = integrate_body_twists(reales[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_odom_rec, p_odom_rec = integrate_body_twists(odom_lin_vel[1:], imu_ang_vel[1:], delta_ts[1:])  # odom_ang_vel
        p_rel_pred_rec, p_pred_rec = integrate_body_twists(predt[1:], imu_ang_vel[1:], delta_ts[1:])

        odom_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_odom_rec,'full')
        pred_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec,'full')


        original_odom_rpe=np.loadtxt(error_file, delimiter=",")
        original_odom_rpe=original_odom_rpe[lag:]
        original_odom_rpe = original_odom_rpe[~np.isnan(original_odom_rpe)]

        T_ws_PRED = np.array(p_pred_rec)
        T_ws_ODOM = np.array(p_odom_rec)
        T_ws_GT = np.array(p_gt_rec)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"Results for the day {mydate.date()}")

        plt.plot(T_ws_GT[:, 0, 3], T_ws_GT[:, 1, 3], label="Ground Truth", color='r', linewidth=1)
        plt.plot(T_ws_PRED[:, 0, 3], T_ws_PRED[:, 1, 3], label="Prediction", color='blue', linewidth=1)
        plt.plot(pos_gt[:, 0] - pos_gt[0, 0], pos_gt[:, 1] - pos_gt[0, 1],linestyle='--', color='orange', label='Raw Trajectory')
        plt.plot(pos_odom[:, 0], pos_odom[:, 1] , linestyle='--',color='turquoise', label='Odom Trajectory')
        #plt.plot(T_ws_ODOM[:, 0, 3], T_ws_ODOM[:, 1, 3], label="Odometry corrected", color='g', linewidth=1)

        plt.scatter(pos_odom[0, 0], pos_odom[0, 1], alpha=0.4, color='turquoise', marker='o')
        plt.scatter(pos_odom[-1, 0], pos_odom[-1, 1], alpha=0.4, color='turquoise', marker='o')
        plt.scatter(pos_gt[1, 0] - pos_gt[0, 0], pos_gt[1, 1] - pos_gt[0, 1], alpha=0.4, color='orange', marker='X')
        plt.scatter(pos_gt[-1, 0] - pos_gt[0, 0], pos_gt[-1, 1] - pos_gt[0, 1], alpha=0.4, color='orange', marker='X')

        plt.legend()
        plt.xlabel("Coord X")
        plt.ylabel('Coord Y')
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(odom_rpe, showfliers=True)#original_odom_rpe
        axs[0].set_title("Odometry RPE from relative position")
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("Prediction RPE from relative position")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(original_odom_rpe, showfliers=True)
        axs[0].set_title("ORIGINAL Odometry RPE from relative position")
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("Prediction RPE from relative position")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        # stats
        media_error = np.mean(original_odom_rpe)
        ORIGINAL_ODOM_abs = np.abs(original_odom_rpe)
        mae = np.mean(ORIGINAL_ODOM_abs)
        max = np.max(ORIGINAL_ODOM_abs)
        num, out = count_outliers(ORIGINAL_ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ORIGINAL_ODOM_abs, 50)
        p90 = np.percentile(ORIGINAL_ODOM_abs, 90)
        p95 = np.percentile(ORIGINAL_ODOM_abs, 95)
        p99 = np.percentile(ORIGINAL_ODOM_abs, 99)

        print("Mean ORIGINAL error ODOM:", media_error)
        print("ORIGINAL Maximum Error:", max)
        print("ORIGINAL MAE ODOM:", mae)
        print("ORIGINAL P50 ODOM:", p50)
        print("ORIGINAL P90 ODOM:", p90)
        print("ORIGINAL P95 ODOM:", p95)
        print("ORIGINAL P99 ODOM:", p99)

        # stats
        media_error = np.mean(odom_rpe)
        ODOM_abs = np.abs(odom_rpe)
        mae = np.mean(ODOM_abs)
        max = np.max(ODOM_abs)
        num, out = count_outliers(ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ODOM_abs, 50)
        p90 = np.percentile(ODOM_abs, 90)
        p95 = np.percentile(ODOM_abs, 95)
        p99 = np.percentile(ODOM_abs, 99)

        print("Mean errorODOM:", media_error)
        print("Maximum Error:", max)
        print("MAE ODOM:", mae)
        print("P50 ODOM:", p50)
        print("P90 ODOM:", p90)
        print("P95 ODOM:", p95)
        print("P99 ODOM:", p99)


        media_error = np.mean(pred_rpe)
        PRED_abs = np.abs(pred_rpe)
        mae = np.mean(PRED_abs)
        max2 = np.max(PRED_abs)

        num, out = count_outliers(PRED_abs)
        print(num, "outliers PRED")

        p50 = np.percentile(PRED_abs, 50)
        p90 = np.percentile(PRED_abs, 90)
        p95 = np.percentile(PRED_abs, 95)
        p99 = np.percentile(PRED_abs, 99)

        print("Mean error PRED:", media_error)
        print("Maximum Error:", max2)
        print("MAE PRED:", mae)
        print("P50 PRED:", p50)
        print("P90 PRED:", p90)
        print("P95 PRED:", p95)
        print("P99 PRED:", p99)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str('Comparison odom Linear Velocity vs Predicted Linear Velocity' + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(ODOM_abs, showfliers=True)
        axs[0].set_title("Odometry Absolute error")
        axs[1].boxplot(PRED_abs, showfliers=True)
        axs[1].set_title("Prediction  Absolute error")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        # stats
        print("Bias ODOM:", np.mean(odom_lin_vel - reales))
        media_error = np.mean(error_ODOM)
        error_ODOM_abs = np.abs(error_ODOM)
        mae = np.mean(error_ODOM_abs)
        max = np.max(error_ODOM_abs)
        num, out = count_outliers(error_ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ODOM_abs, 50)
        p90 = np.percentile(ODOM_abs, 90)
        p95 = np.percentile(ODOM_abs, 95)
        p99 = np.percentile(ODOM_abs, 99)

        print("Mean errorODOM:", media_error)
        print("Maximum Error:", max)
        print("MAE ODOM:", mae)
        print("P50 ODOM:", p50)
        print("P90 ODOM:", p90)
        print("P95 ODOM:", p95)
        print("P99 ODOM:", p99)

        print("Bias PREDICTION:", np.mean(predt - reales))
        media_error = np.mean(error_PRED)
        error_PRED_abs = np.abs(error_PRED)
        mae = np.mean(error_PRED_abs)
        max2 = np.max(error_PRED_abs)

        num, out = count_outliers(error_PRED_abs)
        print(num, "outliers PRED")

        p50 = np.percentile(error_PRED_abs, 50)
        p90 = np.percentile(error_PRED_abs, 90)
        p95 = np.percentile(error_PRED_abs, 95)
        p99 = np.percentile(error_PRED_abs, 99)

        print("Mean error PRED:", media_error)
        print("Maximum Error:", max2)
        print("MAE PRED:", mae)
        print("P50 PRED:", p50)
        print("P90 PRED:", p90)
        print("P95 PRED:", p95)
        print("P99 PRED:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"LINEAR VELOCITY {mydate.date()}")
        plt.plot(reales, label="Ground Truth", color='r', linewidth=1)
        plt.plot(predt, label="Prediction", color='blue', linewidth=1)
        plt.plot(odom_lin_vel,  color='turquoise',label='Odometry')
        plt.legend()
        plt.show(block=True)

def IMPROVEMENT_VISUALIZATION_error(X_t,y_t,pred,title):
    indice = y_t.index

    fechas_unicas = y_t.index.normalize().unique()
    pred_total=pred.copy()
    for mydate in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == mydate)
        predt = pd.Series(pred_total, index=indice)

        horas = indice[mask]
        reales = y_t[mask]
        odom_lin_vel = X_t[mask]['lin_vel_odom_x']
        odom_ang_vel = X_t[mask]['ang_vel_odom_z']
        imu_ang_vel = X_t[mask]['ang_vel_imu_z']
        prederror = predt[mask]
        inferencia=odom_lin_vel.values+prederror.values/10

        if((mydate!=pd.Timestamp('2025-11-03 00:00:00')) & (mydate!=pd.Timestamp('2024-11-28 00:00:00'))):
            color='red'
        elif(mydate!=pd.Timestamp('2025-11-03 00:00:00')):
            color = 'yellow'
        else:
            color = 'blue'

        fecha_str = mydate.date().strftime("%Y-%m-%d")
        path_odom = NEW_FOMO_PATH + f"{fecha_str}_{color}_odom_position.csv"
        path_index = NEW_FOMO_PATH + f"{fecha_str}_{color}_gt_raw_index.csv"
        error_file = NEW_FOMO_PATH + f"{fecha_str}_{color}_error.csv"
        trajectoryGT = file_interface.read_tum_trajectory_file(NEW_FOMO_PATH + f"{fecha_str}_{color}_trajectory.txt")

        pos_odom = np.loadtxt(path_odom, delimiter=",")
        pos_gt = trajectoryGT.positions_xyz
        pos_gt = pos_gt[~np.isnan(pos_gt).any(axis=1)]

        print(odom_ang_vel.shape[0], reales.shape[0], odom_lin_vel.shape[0], inferencia.shape[0])

        delta_ts = horas.diff().values.astype(np.int64) / 1e9
        print(delta_ts.shape[0])

        p_rel_gt_rec, p_gt_rec = integrate_body_twists(reales[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_odom_rec, p_odom_rec = integrate_body_twists(odom_lin_vel[1:], imu_ang_vel[1:], delta_ts[1:])  # odom_ang_vel
        p_rel_pred_rec, p_pred_rec = integrate_body_twists(inferencia[1:], imu_ang_vel[1:], delta_ts[1:])

        odom_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_odom_rec,'full')
        pred_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec,'full')

        original_odom_rpe=np.loadtxt(error_file, delimiter=",")
        original_odom_rpe = original_odom_rpe[~np.isnan(original_odom_rpe)]

        T_ws_PRED = np.array(p_pred_rec)
        T_ws_ODOM = np.array(p_odom_rec)
        T_ws_GT = np.array(p_gt_rec)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"Results for the day {mydate.date()}")
        plt.plot(T_ws_GT[:, 0, 3], T_ws_GT[:, 1, 3], label="Ground Truth", color='r', linewidth=1)
        plt.plot(T_ws_PRED[:, 0, 3], T_ws_PRED[:, 1, 3], label="Prediction", color='blue', linewidth=1)
        plt.plot(pos_gt[:, 0] - pos_gt[0, 0], pos_gt[:, 1] - pos_gt[0, 1],linestyle='--', color='orange', label='Raw Trajectory')
        plt.plot(pos_odom[:, 0], pos_odom[:, 1] , linestyle='--',color='turquoise', label='Odom Trajectory')
        plt.plot(T_ws_ODOM[:, 0, 3], T_ws_ODOM[:, 1, 3], label="Odometry corrected", color='g', linewidth=1)
        #plt.scatter(pos_odom[0, 0], pos_odom[0, 1], alpha=0.4, color='turquoise', marker='o')
        #plt.scatter(pos_odom[-1, 0], pos_odom[-1, 1], alpha=0.4, color='turquoise', marker='o')
        #plt.scatter(pos_gt[1, 0] - pos_gt[0, 0], pos_gt[1, 1] - pos_gt[0, 1], alpha=0.4, color='orange', marker='X')
        #plt.scatter(pos_gt[-1, 0] - pos_gt[0, 0], pos_gt[-1, 1] - pos_gt[0, 1], alpha=0.4, color='orange', marker='X')
        plt.legend()
        plt.xlabel("Coord X")
        plt.ylabel('Coord Y')
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(odom_rpe, showfliers=True)
        axs[0].set_title("Odometry RPE from relative position")
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("Prediction RPE from relative position")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(original_odom_rpe, showfliers=True)
        axs[0].set_title("ORIGINAL Odometry RPE from relative position")
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("Prediction RPE from relative position")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        # stats
        media_error = np.mean(original_odom_rpe)
        ORIGINAL_ODOM_abs = np.abs(original_odom_rpe)
        mae = np.mean(ORIGINAL_ODOM_abs)
        max = np.max(ORIGINAL_ODOM_abs)
        num, out = count_outliers(ORIGINAL_ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ORIGINAL_ODOM_abs, 50)
        p90 = np.percentile(ORIGINAL_ODOM_abs, 90)
        p95 = np.percentile(ORIGINAL_ODOM_abs, 95)
        p99 = np.percentile(ORIGINAL_ODOM_abs, 99)

        print("Mean ORIGINAL error ODOM:", media_error)
        print("ORIGINAL Maximum Error:", max)
        print("ORIGINAL MAE ODOM:", mae)
        print("ORIGINAL P50 ODOM:", p50)
        print("ORIGINAL P90 ODOM:", p90)
        print("ORIGINAL P95 ODOM:", p95)
        print("ORIGINAL P99 ODOM:", p99)

        # stats
        media_error = np.mean(odom_rpe)
        ODOM_abs = np.abs(odom_rpe)
        mae = np.mean(ODOM_abs)
        max = np.max(ODOM_abs)
        num, out = count_outliers(ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ODOM_abs, 50)
        p90 = np.percentile(ODOM_abs, 90)
        p95 = np.percentile(ODOM_abs, 95)
        p99 = np.percentile(ODOM_abs, 99)

        print("Mean errorODOM:", media_error)
        print("Maximum Error:", max)
        print("MAE ODOM:", mae)
        print("P50 ODOM:", p50)
        print("P90 ODOM:", p90)
        print("P95 ODOM:", p95)
        print("P99 ODOM:", p99)

        media_error = np.mean(pred_rpe)
        PRED_abs = np.abs(pred_rpe)
        mae = np.mean(PRED_abs)
        max2 = np.max(PRED_abs)

        num, out = count_outliers(PRED_abs)
        print(num, "outliers PRED")

        p50 = np.percentile(PRED_abs, 50)
        p90 = np.percentile(PRED_abs, 90)
        p95 = np.percentile(PRED_abs, 95)
        p99 = np.percentile(PRED_abs, 99)

        print("Mean error PRED:", media_error)
        print("Maximum Error:", max2)
        print("MAE PRED:", mae)
        print("P50 PRED:", p50)
        print("P90 PRED:", p90)
        print("P95 PRED:", p95)
        print("P99 PRED:", p99)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"LINEAR VELOCITY {mydate.date()}")
        plt.plot(reales.values, label="Ground Truth",alpha=0.5, color='r', linewidth=1)
        plt.plot(inferencia, label="Prediction", alpha=0.5,color='blue', linewidth=1)
        plt.plot(odom_lin_vel.values,  color='turquoise',alpha=0.5,label='Odometry')
        plt.legend()
        plt.show(block=True)

def IMPROVEMENT_VISUALIZATION_error_lag(X_t,y_t,pred,features,lab_target,title):
    indice = y_t.index

    lag = len(features)

    '''
    # inference
    error_pred = model.predict(X_test) / scale
    v_pred = v_odom + error_pred
    '''

    fechas_unicas = y_t.index.normalize().unique()
    pred_total=pred.copy()
    for mydate in fechas_unicas:
        mask = (pd.to_datetime(indice[lag:]).normalize() == mydate)
        predt = pd.Series(pred_total, index=indice[lag:])

        horas = indice[lag:][mask]
        reales = y_t[lag:][mask]
        odom_lin_vel = X_t[lag:][mask]['lin_vel_odom_x']
        odom_ang_vel = X_t[lag:][mask]['ang_vel_odom_z']
        imu_ang_vel = X_t[lag:][mask]['ang_vel_imu_z']
        prederror = predt[mask]
        inferencia=odom_lin_vel+prederror/10


        if((mydate!=pd.Timestamp('2025-11-03 00:00:00')) & (mydate!=pd.Timestamp('2024-11-28 00:00:00'))):
            color='red'
        elif(mydate!=pd.Timestamp('2025-11-03 00:00:00')):
            color = 'yellow'
        else:
            color = 'blue'

        fecha_str = mydate.date().strftime("%Y-%m-%d")
        path_odom = NEW_FOMO_PATH + f"{fecha_str}_{color}_odom_position.csv"
        path_index = NEW_FOMO_PATH + f"{fecha_str}_{color}_gt_raw_index.csv"
        error_file = NEW_FOMO_PATH + f"{fecha_str}_{color}_error.csv"
        trajectoryGT = file_interface.read_tum_trajectory_file(NEW_FOMO_PATH + f"{fecha_str}_{color}_trajectory.txt")
        trajectoryGT.reduce_to_ids(list(range(lag + 1, trajectoryGT.num_poses)))

        pos_odom = np.loadtxt(path_odom, delimiter=",")
        pos_gt = trajectoryGT.positions_xyz
        pos_gt = pos_gt[~np.isnan(pos_gt).any(axis=1)]

        print(odom_ang_vel.shape[0], reales.shape[0], odom_lin_vel.shape[0], inferencia.shape[0])

        delta_ts = horas.diff().values.astype(np.int64) / 1e9
        print(delta_ts.shape[0])

        p_rel_gt_rec, p_gt_rec = integrate_body_twists(reales[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_odom_rec, p_odom_rec = integrate_body_twists(odom_lin_vel[1:], imu_ang_vel[1:], delta_ts[1:])  # odom_ang_vel
        p_rel_pred_rec, p_pred_rec = integrate_body_twists(inferencia[1:], imu_ang_vel[1:], delta_ts[1:])

        odom_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_odom_rec,'full')
        pred_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec,'full')

        original_odom_rpe=np.loadtxt(error_file, delimiter=",")
        original_odom_rpe = original_odom_rpe[lag:]
        original_odom_rpe = original_odom_rpe[~np.isnan(original_odom_rpe)]

        T_ws_PRED = np.array(p_pred_rec)
        T_ws_ODOM = np.array(p_odom_rec)
        T_ws_GT = np.array(p_gt_rec)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"Results for the day {mydate.date()}")
        plt.plot(T_ws_GT[:, 0, 3], T_ws_GT[:, 1, 3], label="Ground Truth", color='r', linewidth=1)
        plt.plot(T_ws_PRED[:, 0, 3], T_ws_PRED[:, 1, 3], label="Prediction", color='blue', linewidth=1)
        plt.plot(pos_gt[:, 0] - pos_gt[0, 0], pos_gt[:, 1] - pos_gt[0, 1],linestyle='--', color='orange', label='Raw Trajectory')
        plt.plot(pos_odom[:, 0], pos_odom[:, 1] , linestyle='--',color='turquoise', label='Odom Trajectory')
        #plt.plot(T_ws_ODOM[:, 0, 3], T_ws_ODOM[:, 1, 3], label="Odometry corrected", color='g', linewidth=1)
        plt.scatter(pos_odom[0, 0], pos_odom[0, 1], alpha=0.4, color='turquoise', marker='o')
        plt.scatter(pos_odom[-1, 0], pos_odom[-1, 1], alpha=0.4, color='turquoise', marker='o')
        plt.scatter(pos_gt[1, 0] - pos_gt[0, 0], pos_gt[1, 1] - pos_gt[0, 1], alpha=0.4, color='orange', marker='X')
        plt.scatter(pos_gt[-1, 0] - pos_gt[0, 0], pos_gt[-1, 1] - pos_gt[0, 1], alpha=0.4, color='orange', marker='X')
        plt.legend()
        plt.xlabel("Coord X")
        plt.ylabel('Coord Y')
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(odom_rpe, showfliers=True)
        axs[0].set_title("Odometry RPE from relative position")
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("Prediction RPE from relative position")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"BOXPLOT  {mydate.date()}")
        axs[0].boxplot(original_odom_rpe, showfliers=True)
        axs[0].set_title("ORIGINAL Odometry RPE from relative position")
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("Prediction RPE from relative position")
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        # stats
        media_error = np.mean(original_odom_rpe)
        ORIGINAL_ODOM_abs = np.abs(original_odom_rpe)
        mae = np.mean(ORIGINAL_ODOM_abs)
        max = np.max(ORIGINAL_ODOM_abs)
        num, out = count_outliers(ORIGINAL_ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ORIGINAL_ODOM_abs, 50)
        p90 = np.percentile(ORIGINAL_ODOM_abs, 90)
        p95 = np.percentile(ORIGINAL_ODOM_abs, 95)
        p99 = np.percentile(ORIGINAL_ODOM_abs, 99)

        print("Mean ORIGINAL error ODOM:", media_error)
        print("ORIGINAL Maximum Error:", max)
        print("ORIGINAL MAE ODOM:", mae)
        print("ORIGINAL P50 ODOM:", p50)
        print("ORIGINAL P90 ODOM:", p90)
        print("ORIGINAL P95 ODOM:", p95)
        print("ORIGINAL P99 ODOM:", p99)

        # stats
        media_error = np.mean(odom_rpe)
        ODOM_abs = np.abs(odom_rpe)
        mae = np.mean(ODOM_abs)
        max = np.max(ODOM_abs)
        num, out = count_outliers(ODOM_abs)
        print(num, "outliers ODOM")

        p50 = np.percentile(ODOM_abs, 50)
        p90 = np.percentile(ODOM_abs, 90)
        p95 = np.percentile(ODOM_abs, 95)
        p99 = np.percentile(ODOM_abs, 99)

        print("Mean errorODOM:", media_error)
        print("Maximum Error:", max)
        print("MAE ODOM:", mae)
        print("P50 ODOM:", p50)
        print("P90 ODOM:", p90)
        print("P95 ODOM:", p95)
        print("P99 ODOM:", p99)

        media_error = np.mean(pred_rpe)
        PRED_abs = np.abs(pred_rpe)
        mae = np.mean(PRED_abs)
        max2 = np.max(PRED_abs)

        num, out = count_outliers(PRED_abs)
        print(num, "outliers PRED")

        p50 = np.percentile(PRED_abs, 50)
        p90 = np.percentile(PRED_abs, 90)
        p95 = np.percentile(PRED_abs, 95)
        p99 = np.percentile(PRED_abs, 99)

        print("Mean error PRED:", media_error)
        print("Maximum Error:", max2)
        print("MAE PRED:", mae)
        print("P50 PRED:", p50)
        print("P90 PRED:", p90)
        print("P95 PRED:", p95)
        print("P99 PRED:", p99)


        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=True)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"LINEAR VELOCITY {mydate.date()}")
        plt.plot(reales, label="Ground Truth", color='r', linewidth=1)
        plt.plot(inferencia, label="Prediction", color='blue', linewidth=1)
        plt.plot(odom_lin_vel,  color='turquoise',label='Odometry')
        plt.legend()
        plt.show(block=True)
















