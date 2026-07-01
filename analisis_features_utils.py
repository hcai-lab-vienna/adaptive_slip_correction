import pandas as pd
pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor
from sklearn.metrics import  r2_score

import os

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
CONDITIONS_DATE=[ ("2025-04-15",1),("2025-11-03",3),("2024-11-28",1),("2025-06-26",2),("2025-09-24",2),("2024-11-21",2),("2025-05-28",2),("2025-08-20",2),("2025-10-14",2)]

# ============================================================
# LOAD and PREPROCESS FUNCTIONS
# ============================================================
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
    print('DRY-train-test: Kolmogorov Smirnov Test Results:', ks_clear_train_test)

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]
    LABELS_ENTRENAMIENTO = [y_snow_road, y_clear_road, y_clear_raining]
    INDICES = [y_test_snow_road_index, y_test_clear_road_index, y_test_clear_raining_index]
    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]

    results = []
    prediction = []
    conditions = ["snowy_road", "dry_road", "rainy_road"]
    for est1, X_train, y_train, X_test, y_test in zip(
            conditions,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Training conditions: {est1}")

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X_train, y_train)

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

            prediction_xgb=model.predict(X_test)
            prediction.append(prediction_xgb)

            mae_pred = mean_absolute_error(y_test, prediction_xgb)
            r2=r2_score(y_test, prediction_xgb)

            print(f"Time series: Real {est1} vs Prediction {est2}")
            print(f"MAE : {mae_pred:.4f}")

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
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")#viridis
    plt.title(" MAE (train vs test)")
    plt.show()

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

def experiment_RQ1(input,test,features,LABEL_SV):
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

    print('SNOW mean & median ', mediasnow, medianasnow)
    print('DRY mean & median ', mediaclear, medianaclear)
    print('RAIN mean & median ', mediarain, medianarain)

    features_cor=  features + [LABEL_SV]

    corr_snow = df_snow_road[features_cor].corr(numeric_only=True)
    corr_clear = df_clear_road[features_cor].corr(numeric_only=True)
    corr_rain = df_clear_raining[features_cor].corr(numeric_only=True)
    mask_snow = np.abs(corr_snow) < 0.3
    mask_clear = np.abs(corr_clear) < 0.3
    mask_rain = np.abs(corr_rain) < 0.3

    corr_check = corr_rain.copy()
    corr_check.insert(0, "feature_name", corr_check.index)

    corr_check2 = corr_clear.copy()
    corr_check2.insert(0, "feature_name", corr_check2.index)

    corr_check3 = corr_snow.copy()
    corr_check3.insert(0, "feature_name", corr_check3.index)

    corr_target_snow = corr_snow[LABEL_SV].dropna()
    corr_target_clear = corr_clear[LABEL_SV].dropna()
    corr_target_rain = corr_rain[LABEL_SV].dropna()

    ks_snow_rain = ks_2samp(df_snow_road[LABEL_SV], df_clear_raining[LABEL_SV])
    ks_snow_clear = ks_2samp(df_snow_road[LABEL_SV], df_clear_road[LABEL_SV])
    ks_clear_rain = ks_2samp(df_clear_road[LABEL_SV], df_clear_raining[LABEL_SV])

    print('snow-rain: Kolmogorov Smirnov Test Results:', ks_snow_rain)
    print('snow-dry: Kolmogorov Smirnov Test Results:', ks_snow_clear)
    print('dry-rain: Kolmogorov Smirnov Test Results:', ks_clear_rain)

    plt.figure(figsize=(10, 8))
    plot_cdf(df_snow_road[LABEL_SV], "snow")
    plot_cdf(df_clear_road[LABEL_SV], "dry")
    plot_cdf(df_clear_raining[LABEL_SV], "rain")
    plt.legend()
    plt.title("CDF comparison")
    plt.xlabel(LABEL_SV)
    plt.ylabel("F(x)")
    plt.grid()
    plt.show(block=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].set_title("Histogram")
    sns.histplot(y_snow_road, alpha=0.5, bins=1000, color='r', label='Snow', ax=axes[0])
    axes[0].axvline(mediasnow, color='red', linestyle='solid', linewidth=2, label=f'Mean snow: {mediasnow:.2f}')
    axes[0].axvline(medianasnow, color='red', linestyle='dashed', linewidth=2, label=f'Median snow: {medianasnow:.2f}')

    sns.histplot(y_clear_road, alpha=0.5, bins=1000, color='g', label='Dry', ax=axes[0])
    axes[0].axvline(mediaclear, color='k', linestyle='solid', linewidth=2, label=f'Mean dry: {mediaclear:.2f}')
    axes[0].axvline(medianaclear, color='k', linestyle='dashed', linewidth=2, label=f'Median dry: {medianaclear:.2f}')

    sns.histplot(y_clear_raining, alpha=0.5, bins=1000, color='b', label='Rain', ax=axes[0])
    axes[0].axvline(mediarain, color='yellow', linestyle='solid', linewidth=2, label=f'Mean rain: {mediarain:.2f}')
    axes[0].axvline(medianarain, color='yellow', linestyle='dashed', linewidth=2,
                    label=f'Median rain: {medianarain:.2f}')
    axes[0].legend()
    axes[0].set_title("Histograms")

    sns.kdeplot(df_snow_road[LABEL_SV], label="snow", color='r', fill=True, ax=axes[1])
    sns.kdeplot(df_clear_road[LABEL_SV], label="dry", color='g', fill=True, ax=axes[1])
    sns.kdeplot(df_clear_raining[LABEL_SV], label="rain", color='b', fill=True, ax=axes[1])
    axes[1].legend()
    axes[1].set_title("KDE Distributions")

    plt.tight_layout()
    plt.show(block=True)

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
    plt.title("DRY ROAD - Correlations ")
    plt.show(block=True)

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
    plt.title("SNOW ROAD - Correlations ")
    plt.show(block=True)

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
    plt.title("RAIN ROAD - Correlations ")
    plt.show(block=True)

    # plt.figure(figsize=(8, 6))
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
    plt.title("SNOW  - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show(block=True)

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
    plt.title("RAIN - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show(block=True)

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
    plt.title("DRY - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show(block=True)

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
    plt.xticks(ticks=[0, 1, 2], labels=["Snow", "Dry", "Rain"])

    plt.title("Distribution by condition")
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
    plt.xticks(ticks=[0, 1, 2], labels=["Snow", "Dry", "Rain"])
    plt.title("Boxplot real distribution")
    plt.show(block=True)

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
    plt.xticks(ticks=[0, 1, 2], labels=["Snow", "Dry", "Rain"])
    plt.title("Kernel density–based distribution of SV by condition.")  # SV distribution by  condition
    plt.show(block=True)

    summary = df_all.groupby("Soil_type")[LABEL_SV].describe()
    print(summary)

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    ks_snow_train_test = ks_2samp(df_snow_road[LABEL_SV], df_TEST_snow_road[LABEL_SV])
    ks_rain_train_test = ks_2samp(df_clear_raining[LABEL_SV], df_TEST_clear_raining[LABEL_SV])
    ks_clear_train_test = ks_2samp(df_clear_road[LABEL_SV], df_TEST_clear_road[LABEL_SV])

    print('SNOW-train-test: Kolmogorov Smirnov Test Results:', ks_snow_train_test)
    print('RAIN-train-test: Kolmogorov Smirnov Test Results:', ks_rain_train_test)
    print('DRY-train-test: Kolmogorov Smirnov Test Results:', ks_clear_train_test)

def experiment_RQ2(input,test,features,LABEL_SV):
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

    print('SNOW mean & median ', mediasnow, medianasnow)
    print('DRY mean & median ', mediaclear, medianaclear)
    print('RAIN mean & median ', mediarain, medianarain)

    features_cor = features + [LABEL_SV]

    corr_snow = df_snow_road[features_cor].corr(numeric_only=True)
    corr_clear = df_clear_road[features_cor].corr(numeric_only=True)
    corr_rain = df_clear_raining[features_cor].corr(numeric_only=True)
    mask_snow = np.abs(corr_snow) < 0.3
    mask_clear = np.abs(corr_clear) < 0.3
    mask_rain = np.abs(corr_rain) < 0.3

    corr_check = corr_rain.copy()
    corr_check.insert(0, "feature_name", corr_check.index)

    corr_check2 = corr_clear.copy()
    corr_check2.insert(0, "feature_name", corr_check2.index)

    corr_check3 = corr_snow.copy()
    corr_check3.insert(0, "feature_name", corr_check3.index)

    corr_target_snow = corr_snow[LABEL_SV].dropna()
    corr_target_clear = corr_clear[LABEL_SV].dropna()
    corr_target_rain = corr_rain[LABEL_SV].dropna()

    ks_snow_rain = ks_2samp(df_snow_road[LABEL_SV], df_clear_raining[LABEL_SV])
    ks_snow_clear = ks_2samp(df_snow_road[LABEL_SV], df_clear_road[LABEL_SV])
    ks_clear_rain = ks_2samp(df_clear_road[LABEL_SV], df_clear_raining[LABEL_SV])

    print('snow-rain: Kolmogorov Smirnov Test Results:', ks_snow_rain)
    print('snow-dry: Kolmogorov Smirnov Test Results:', ks_snow_clear)
    print('dry-rain: Kolmogorov Smirnov Test Results:', ks_clear_rain)

    plt.figure(figsize=(10, 8))
    plot_cdf(df_snow_road[LABEL_SV], "snow")
    plot_cdf(df_clear_road[LABEL_SV], "dry")
    plot_cdf(df_clear_raining[LABEL_SV], "rain")
    plt.legend()
    plt.title("CDF comparison")
    plt.xlabel(LABEL_SV)
    plt.ylabel("F(x)")
    plt.grid()
    plt.show(block=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].set_title("Histogram")
    sns.histplot(y_snow_road, alpha=0.5, bins=1000, color='r', label='Snow', ax=axes[0])
    axes[0].axvline(mediasnow, color='red', linestyle='solid', linewidth=2, label=f'Mean snow: {mediasnow:.2f}')
    axes[0].axvline(medianasnow, color='red', linestyle='dashed', linewidth=2, label=f'Median snow: {medianasnow:.2f}')

    sns.histplot(y_clear_road, alpha=0.5, bins=1000, color='g', label='Dry', ax=axes[0])
    axes[0].axvline(mediaclear, color='k', linestyle='solid', linewidth=2, label=f'Mean dry: {mediaclear:.2f}')
    axes[0].axvline(medianaclear, color='k', linestyle='dashed', linewidth=2, label=f'Median dry: {medianaclear:.2f}')

    sns.histplot(y_clear_raining, alpha=0.5, bins=1000, color='b', label='Rain', ax=axes[0])
    axes[0].axvline(mediarain, color='yellow', linestyle='solid', linewidth=2, label=f'Mean rain: {mediarain:.2f}')
    axes[0].axvline(medianarain, color='yellow', linestyle='dashed', linewidth=2,
                    label=f'Median rain: {medianarain:.2f}')
    axes[0].legend()
    axes[0].set_title("Histograms")

    sns.kdeplot(df_snow_road[LABEL_SV], label="snow", color='r', fill=True, ax=axes[1])
    sns.kdeplot(df_clear_road[LABEL_SV], label="dry", color='g', fill=True, ax=axes[1])
    sns.kdeplot(df_clear_raining[LABEL_SV], label="rain", color='b', fill=True, ax=axes[1])
    axes[1].legend()
    axes[1].set_title("KDE Distributions")

    plt.tight_layout()
    plt.show(block=True)

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
    plt.title("DRY ROAD - Correlations ")
    plt.show(block=True)

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
    plt.title("SNOW ROAD - Correlations ")
    plt.show(block=True)

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
    plt.title("RAIN ROAD - Correlations ")
    plt.show(block=True)

    # plt.figure(figsize=(8, 6))
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
    plt.title("SNOW  - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show(block=True)

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
    plt.title("RAIN - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show(block=True)

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
    plt.title("DRY - Correlations ")
    plt.ylabel("Features")
    plt.xlabel("")
    plt.show(block=True)

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
    plt.xticks(ticks=[0, 1, 2], labels=["Snow", "Dry", "Rain"])

    plt.title("Distribution by condition")
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
    plt.xticks(ticks=[0, 1, 2], labels=["Snow", "Dry", "Rain"])
    plt.title("Boxplot real distribution")
    plt.show(block=True)

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
    plt.xticks(ticks=[0, 1, 2], labels=["Snow", "Dry", "Rain"])
    plt.title("Kernel density–based distribution of SV by condition.")  # SV distribution by  condition
    plt.show(block=True)

    summary = df_all.groupby("Soil_type")[LABEL_SV].describe()
    print(summary)

    df_TEST_snow_road = test[1]
    df_TEST_clear_road = test[2]
    df_TEST_clear_raining = test[3]

    ks_snow_train_test = ks_2samp(df_snow_road[LABEL_SV], df_TEST_snow_road[LABEL_SV])
    ks_rain_train_test = ks_2samp(df_clear_raining[LABEL_SV], df_TEST_clear_raining[LABEL_SV])
    ks_clear_train_test = ks_2samp(df_clear_road[LABEL_SV], df_TEST_clear_road[LABEL_SV])

    print('SNOW-train-test: Kolmogorov Smirnov Test Results:', ks_snow_train_test)
    print('RAIN-train-test: Kolmogorov Smirnov Test Results:', ks_rain_train_test)
    print('DRY-train-test: Kolmogorov Smirnov Test Results:', ks_clear_train_test)

def experiment_RQ3(XTEST,YTEST,pred3,res1,pred1,predwithout):
    predwithout = predwithout[np.argsort(YTEST.index)]

    indice = YTEST.index
    fechas_unicas = YTEST.index.normalize().unique()
    print(fechas_unicas)
    pred3_total = pred3.copy()
    pred1_total = pred1.copy()
    predw_total = predwithout.copy()

    for mydate in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == mydate)
        pred3t = pd.Series(pred3_total, index=indice)
        pred1t = pd.Series(pred1_total, index=indice)
        predw = pd.Series(predw_total, index=indice)

        horas = indice[mask]
        reales = YTEST[mask]
        odom_lin_vel = XTEST[mask]['lin_vel_odom_x']
        odom_ang_vel = XTEST[mask]['ang_vel_odom_z']
        imu_ang_vel = XTEST[mask]['ang_vel_imu_z']
        prederror3 = pred3t[mask]
        prederror1 = pred1t[mask]
        prederrorw = predw[mask]
        inferencia3 = odom_lin_vel.values + prederror3.values
        inferencia1 = odom_lin_vel.values + prederror1.values
        inferenciaw = odom_lin_vel.values + prederrorw.values

        delta_ts = horas.diff().values.astype(np.int64) / 1e9
        print(delta_ts.shape[0])

        p_rel_gt_rec, p_gt_rec = integrate_body_twists(reales[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_odom_rec, p_odom_rec = integrate_body_twists(odom_lin_vel[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_pred_rec3, p_pred_rec3 = integrate_body_twists(inferencia3[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_pred_rec1, p_pred_rec1 = integrate_body_twists(inferencia1[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_pred_recw, p_pred_recw = integrate_body_twists(inferenciaw[1:], imu_ang_vel[1:], delta_ts[1:])

        odom_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_odom_rec, 'full')
        pred_rpe3 = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec3, 'full')
        pred_rpe1 = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec1, 'full')
        pred_rpew = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_recw, 'full')

        T_ws_PRED3 = np.array(p_pred_rec3)
        T_ws_PRED1 = np.array(p_pred_rec1)
        T_ws_PREDw = np.array(p_pred_recw)
        T_ws_ODOM = np.array(p_odom_rec)
        T_ws_GT = np.array(p_gt_rec)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        titulo = str('Comparison 3 models for the Linear Velocity Prediction ' + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        axs[0].set_title("Trajectories")
        axs[0].plot(T_ws_GT[:, 0, 3], T_ws_GT[:, 1, 3], label="Recon. GT ",alpha=0.5, color='r', linewidth=1)
        axs[0].plot(T_ws_PREDw[:, 0, 3], T_ws_PREDw[:, 1, 3], label="Recon. Pred. without SV", alpha=0.5,color='green', linewidth=1)
        axs[0].plot(T_ws_PRED3[:, 0, 3], T_ws_PRED3[:, 1, 3], label="Recon. Pred. 3SV",alpha=0.9, color='turquoise', linewidth=1)
        axs[0].plot(T_ws_PRED1[:, 0, 3], T_ws_PRED1[:, 1, 3], label="Recon. Pred. 1SV", alpha=0.5, color='blue',
                    linewidth=1, linestyle='--')
        axs[0].legend(loc='upper right')
        axs[0].set_xlabel("Coord X (m)")
        axs[0].set_ylabel('Coord Y (m)')
        axs[1].boxplot([pred_rpew, pred_rpe1, pred_rpe3], showfliers=True)
        axs[1].set_title("RPE from relative position")
        axs[1].set_ylabel('RPE (m)')
        axs[1].set_xticklabels(["Model without SV","1SV Model","3SV Model"])
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        reduction3 = 100 * (np.median(odom_rpe) - np.median(pred_rpe3)) / np.median(odom_rpe)
        reduction1 = 100 * (np.median(odom_rpe) - np.median(pred_rpe1)) / np.median(odom_rpe)
        reductionw = 100 * (np.median(odom_rpe) - np.median(pred_rpew)) / np.median(odom_rpe)

        print(str(mydate),
              ': The 3SV method achieves a reduction of % in median RPE with respect to corrected odometry reconstructed ',
              reduction3)
        print(str(mydate),
              ': The 1SV method achieves a reduction of % in median RPE with respect to corrected odometry reconstructed ',
              reduction1)
        print(str(mydate),
              ': The without method achieves a reduction of % in median RPE with respect to corrected odometry reconstructed ',
              reductionw)

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

        media_error = np.mean(pred_rpe3)
        PRED_abs = np.abs(pred_rpe3)
        mae = np.mean(PRED_abs)
        max2 = np.max(PRED_abs)

        num, out = count_outliers(PRED_abs)
        print('\n 3SV METHOD: ',num, "outliers PRED")

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

        media_error = np.mean(pred_rpe1)
        PRED_abs = np.abs(pred_rpe1)
        mae = np.mean(PRED_abs)
        max2 = np.max(PRED_abs)

        num, out = count_outliers(PRED_abs)
        print('\n 1SV METHOD: ', num, "outliers PRED")

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

        media_error = np.mean(pred_rpew)
        PRED_abs = np.abs(pred_rpew)
        mae = np.mean(PRED_abs)
        max2 = np.max(PRED_abs)

        num, out = count_outliers(PRED_abs)
        print('\n WITHOUT METHOD: ', num, "outliers PRED")

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
        titulo = str('ERROR Comparison 3 Models for the Linear velocity Prediction ' + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"LINEAR VELOCITY {mydate.date()}")
        plt.plot(reales.values, label="Ground Truth", alpha=0.5, color='r', linewidth=1)
        plt.plot(inferenciaw, label="Without SV Pred.", alpha=0.5, color='green', linewidth=1)
        plt.plot(inferencia3, label="3SV Pred.", alpha=0.9, color='turquoise', linewidth=1)
        plt.plot(inferencia1, label="1SVPred.", alpha=0.5, color='blue', linewidth=1, linestyle='--')
        plt.legend()
        plt.show(block=True)
def extract_xy(T_ws):
    return np.column_stack((T_ws[:, 0, 3], T_ws[:, 1, 3]))
def Save_results(path,XTEST,YTEST,pred3,res1,pred1,predwithout):
    predwithout = predwithout[np.argsort(YTEST.index)]

    indice = YTEST.index
    fechas_unicas = YTEST.index.normalize().unique()
    print(fechas_unicas)
    pred3_total = pred3.copy()
    pred1_total = pred1.copy()
    predw_total = predwithout.copy()

    all_days_results = []
    for mydate in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == mydate)
        pred3t = pd.Series(pred3_total, index=indice)
        pred1t = pd.Series(pred1_total, index=indice)
        predw = pd.Series(predw_total, index=indice)

        horas = indice[mask]
        indice_final = horas[1:]
        reales = YTEST[mask]
        odom_lin_vel = XTEST[mask]['lin_vel_odom_x']
        odom_ang_vel = XTEST[mask]['ang_vel_odom_z']
        imu_ang_vel = XTEST[mask]['ang_vel_imu_z']
        prederror3 = pred3t[mask]
        prederror1 = pred1t[mask]
        prederrorw = predw[mask]
        inferencia3 = odom_lin_vel.values + prederror3.values
        inferencia1 = odom_lin_vel.values + prederror1.values
        inferenciaw = odom_lin_vel.values + prederrorw.values

        delta_ts = horas.diff().values.astype(np.int64) / 1e9
        print(delta_ts.shape[0])

        p_rel_gt_rec, p_gt_rec = integrate_body_twists(reales[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_odom_rec, p_odom_rec = integrate_body_twists(odom_lin_vel[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_pred_rec3, p_pred_rec3 = integrate_body_twists(inferencia3[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_pred_rec1, p_pred_rec1 = integrate_body_twists(inferencia1[1:], imu_ang_vel[1:], delta_ts[1:])
        p_rel_pred_recw, p_pred_recw = integrate_body_twists(inferenciaw[1:], imu_ang_vel[1:], delta_ts[1:])

        odom_rpe = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_odom_rec, 'full')
        pred_rpe3 = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec3, 'full')
        pred_rpe1 = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_rec1, 'full')
        pred_rpew = compute_rpe_from_rel_pose(p_rel_gt_rec, p_rel_pred_recw, 'full')

        T_ws_PRED3 = np.array(p_pred_rec3)
        T_ws_PRED1 = np.array(p_pred_rec1)
        T_ws_PREDw = np.array(p_pred_recw)
        T_ws_ODOM = np.array(p_odom_rec)
        T_ws_GT = np.array(p_gt_rec)

        pos_gt = extract_xy(T_ws_GT[1:])
        pos_odom = extract_xy(T_ws_ODOM[1:])
        pos_pred3 = extract_xy(T_ws_PRED3[1:])
        pos_pred1 = extract_xy(T_ws_PRED1[1:])
        pos_predw = extract_xy(T_ws_PREDw[1:])

        df_dia = pd.DataFrame({
            # Positions Ground Truth
            'gt_x': pos_gt[:, 0], 'gt_y': pos_gt[:, 1],
            'Vel_GT': reales.values[1:],

            # Positions & RPE Odom corrected
            'odom_x': pos_odom[:, 0], 'odom_y': pos_odom[:, 1],
            'rpe_odom': odom_rpe,
            'vel_odom': odom_lin_vel.values[1:],

            # Positions & RPE Method 3SV
            'pred3_x': pos_pred3[:, 0], 'pred3_y': pos_pred3[:, 1],
            'rpe_pred3': pred_rpe3,
            'inf_pred3': inferencia3[1:],

            # Positions & Errors Method 1SV
            'pred1_x': pos_pred1[:, 0], 'pred1_y': pos_pred1[:, 1],
            'rpe_pred1': pred_rpe1,
            'inf_pred1': inferencia1[1:],

            # Positions & RPE Method W
            'predw_x': pos_predw[:, 0], 'predw_y': pos_predw[:, 1],
            'rpe_predw': pred_rpew,
            'inf_w': inferenciaw[1:],
        }, index=indice_final)

        all_days_results.append(df_dia)

    df_final_total = pd.concat(all_days_results)
    filename = path+"comprehensive_analysis_of_trajectories.csv"

    df_final_total.to_csv(filename, index=True, index_label='datetime', decimal='.', encoding='utf-8')
    print(f"✅ All Every day, they have been exported to: {filename}")
    print(df_final_total.head())

def load_trajectory_results(filepath):
    try:
        df = pd.read_csv(filepath, index_col='datetime', parse_dates=True, decimal='.')
        df = df.sort_index()

        print(f"✅ Data successfully loaded from {filepath}")
        print(f"📅 Time range: {df.index.min()} hasta {df.index.max()}")
        print(f"📊 Total number of samples: {len(df)}")

        return df

    except FileNotFoundError:
        print(f"❌ Error: The file {filepath} does not exist.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error while loading the file: {e}")
        return None

def count_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return len(outliers), outliers

def ONE_modelelastic(X,y,X_t,y_t,features,LABEL_SV):
    X_train = X[features].values
    y_train = ( y.values - X['lin_vel_odom_x'])#10 *

    model = XGBRegressor(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42

    )

    model.fit(X_train, y_train)

    X_test = X_t[features].values
    y_test = (y_t.values - X_t['lin_vel_odom_x'])#10 *
    indice = y_t.index

    prediction=model.predict(X_test)

    fechas_unicas = y_t.index.normalize().unique()
    results=[]
    cont=0

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

        '''fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot without SV Prediction', fontsize=16)
        plt.title(f"{LABEL_SV}   Results for the day {fecha.date()}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)
        plt.ylabel("Error")
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

def predict_XGB_dataset(df_dict,features,models,dataset_name):
    results = {}
    conditions = ["snow_road", "dry_road", "rainy_road"]
    for cond_data in conditions:
        df = df_dict[cond_data]
        X = df[features].values
        y = df["SV"].values

        df_result = pd.DataFrame(index=df.index)

        for cond_model in conditions:
            model = models[cond_model]
            pred = model.predict(X)
            df_result[f"SV_{cond_model}_pred"] = pred

        df_result["SV_real"] = y
        results[cond_data] = df_result

        for cond_model in conditions:
            mae = mean_absolute_error(
                df_result["SV_real"],
                df_result[f"SV_{cond_model}_pred"]
            )
            print(f"{dataset_name} | Data={cond_data} | Model={cond_model} | MAE={mae:.4f}")

    return results
def predict_1XGB_dataset(df,df_y,features,model,dataset_name):
    X = df[features].values
    y = df_y.values
    df_result = pd.DataFrame(index=df.index)

    pred = model.predict(X)
    df_result[f"SV_pred"] = pred

    df_result["SV_real"] = y

    mae = mean_absolute_error(
        df_result["SV_real"],
        df_result[f"SV_pred"]
    )
    print(f"{dataset_name}  | MAE={mae:.4f}")

    return df_result

def train_XGB_models(input,features):
    train_dfs = {
        "snow_road": input[1],
        "dry_road": input[2],
        "rainy_road": input[3]
    }

    models = {}

    conditions = ["snow_road", "dry_road", "rainy_road"]
    for cond in conditions:
        df_train = train_dfs[cond]

        X_train = df_train[features].values
        y_train = df_train["SV"].values

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X_train, y_train)
        models[cond] = model

    return models
def train_1XGB_models(input,features):
    train_dfs = {
        "snow_road": input[1],
        "dry_road": input[2],
        "rainy_road": input[3]
    }

    models = {}

    conditions = ["snow_road", "dry_road", "rainy_road"]
    for cond in conditions:
        df_train = train_dfs[cond]

        X_train = df_train[features].values
        y_train = df_train["SV"].values

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X_train, y_train)
        models[cond] = model

    return models

def model_Slip_index(input,test,features):
    print("\n--- Three slip index calculating ---")
    my_XGB_models= train_XGB_models(input, features)
    train_dfs = {
        "snow_road": input[1],
        "dry_road": input[2],
        "rainy_road": input[3]
    }

    test_dfs = {
        "snow_road": test[1],
        "dry_road": test[2],
        "rainy_road": test[3]
    }

    print('XGB')
    train_XGBresults = predict_XGB_dataset(train_dfs, features, my_XGB_models,  "TRAIN")

    print('XGB')
    test_XGB_results = predict_XGB_dataset(test_dfs, features, my_XGB_models, "TEST")

    return  train_XGBresults,test_XGB_results
def model_one_Slip_index(dftrain,dfest,features):
    print("\n--- Three slip index calculating ---")

    X_train = dftrain[features].values
    y_train = dftrain["SV"].values

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)
    train_XGBresults = predict_1XGB_dataset(dftrain[features],dftrain["SV"], features, model, "TRAIN")
    test_XGBresults = predict_1XGB_dataset(dfest[features],dfest["SV"], features, model, "TEST")

    return  train_XGBresults,test_XGBresults

def conditional_model(Xej,y,Xej_t,y_t,features,LABEL_SV):
    lista = features
    X = Xej[lista].copy()
    X_t = Xej_t[lista].copy()

    X_train = X[lista]
    y_train = (y - X['lin_vel_odom_x'])

    model=XGBRegressor(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42

    )
    model.fit(X_train, y_train)

    X_test = X_t[lista]
    y_test = (y_t - X_t['lin_vel_odom_x'])
    indice = y_t.index

    prediction =model.predict(X_test)

    y_t.index = pd.to_datetime(y_t.index)
    fechas_unicas = y_t.index.normalize().unique()
    results=[]
    cont=0

    pred_total = prediction.copy()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == fecha)
        reales = y_test[mask]
        prederror = pred_total[mask]

        odom_lin_vel = X_t[mask]['lin_vel_odom_x']
        inferencia = odom_lin_vel + prederror #/10

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

        '''fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"{LABEL_SV}   Results for the day {fecha.date()}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)

        plt.ylabel("Error")
        plt.show(block=True)'''

        cont+=1

    return results,prediction
def conditional_1model(Xej,y,Xej_t,y_t,features,LABEL_SV):
    lista = features
    X = Xej[lista].copy()
    X_t = Xej_t[lista].copy()

    X_train = X[lista]
    y_train = (y - X['lin_vel_odom_x'])

    model=XGBRegressor(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)

    X_test = X_t[lista]
    y_test = (y_t - X_t['lin_vel_odom_x'])
    indice = y_t.index

    prediction =model.predict(X_test)

    y_t.index = pd.to_datetime(y_t.index)
    fechas_unicas = y_t.index.normalize().unique()
    results=[]
    cont=0

    pred_total = prediction.copy()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice).normalize() == fecha)
        reales = y_test[mask]
        prederror = pred_total[mask]

        odom_lin_vel = X_t[mask]['lin_vel_odom_x']
        inferencia = odom_lin_vel + prederror #/10

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

        '''fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        fig.suptitle('Error Distribution Boxplot', fontsize=16)
        plt.title(f"{LABEL_SV}   Results for the day {fecha.date()}")
        plt.boxplot(errores_abs)
        #plt.boxplot(errores_abs, showfliers=False)

        plt.ylabel("Error")
        plt.show(block=True)'''
        cont+=1

    return results,prediction

def IMPROVEMENT_VISUALIZATION_error(X_t,y_t,pred,title):
    y_t.index = pd.to_datetime(y_t.index)
    indice = y_t.index
    fechas_unicas = y_t.index.normalize().unique()
    print(fechas_unicas)
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
        inferencia=odom_lin_vel.values+prederror.values

        if((mydate!=pd.Timestamp('2025-11-03 00:00:00')) & (mydate!=pd.Timestamp('2024-11-28 00:00:00'))):
            color='red'
            cond='DRY'
        elif(mydate!=pd.Timestamp('2025-11-03 00:00:00')):
            color = 'yellow'
            cond = 'SNOW'
        else:
            color = 'blue'
            cond = 'RAIN'

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

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        axs[0].set_title("Trajectories")
        axs[0].plot(pos_gt[:, 0] - pos_gt[0, 0], pos_gt[:, 1] - pos_gt[0, 1], linestyle='--', color='orange',label='GT Trajectory')
        axs[0].plot(pos_odom[:, 0], pos_odom[:, 1], linestyle='--', color='turquoise', label='Odom Corr Trajectory')
        axs[0].legend(loc='upper right')
        axs[0].set_xlabel("Coord X (m)")
        axs[0].set_ylabel('Coord Y (m)')
        axs[1].boxplot(original_odom_rpe, showfliers=True)
        axs[1].set_xticklabels(["Original Trajectories"])
        axs[1].set_title("RPE from relative position")
        axs[1].set_ylabel('RPE (m)')
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        axs[0].set_title("Trajectories")
        axs[0].plot(T_ws_GT[:, 0, 3], T_ws_GT[:, 1, 3], label="GT Reconstructed", color='r', linewidth=1)
        axs[0].plot(T_ws_PRED[:, 0, 3], T_ws_PRED[:, 1, 3], label="Pred Reconstructed", color='blue', linewidth=1)
        axs[0].legend(loc='upper right')
        axs[0].set_xlabel("Coord X (m)")
        axs[0].set_ylabel('Coord Y (m)')
        axs[1].boxplot(pred_rpe, showfliers=True)
        axs[1].set_title("RPE from relative position")
        axs[1].set_ylabel('RPE (m)')
        axs[1].set_xticklabels(["Reconstructions"])
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        titulo = str(title + str(mydate))
        fig.suptitle(titulo, fontsize=16)
        axs[0].set_title("Trajectories")
        axs[0].plot(T_ws_GT[:, 0, 3], T_ws_GT[:, 1, 3], label="GT Reconstructed", color='r', linewidth=1)
        axs[0].plot(T_ws_ODOM[:, 0, 3], T_ws_ODOM[:, 1, 3],label="Odom Corr Reconstructed", color='g', linewidth=1)
        axs[0].legend(loc='upper right')
        axs[0].set_xlabel("Coord X (m)")
        axs[0].set_ylabel('Coord Y (m)')
        axs[1].boxplot(odom_rpe, showfliers=True)
        axs[1].set_title("RPE from relative position")
        axs[1].set_ylabel('RPE (m)')
        axs[1].set_xticklabels(["Reconstructions"])
        axs[1].tick_params(labelleft=True)
        plt.tight_layout()
        plt.show(block=True)

        reduction = 100 * (np.median(odom_rpe) - np.median(pred_rpe)) / np.median(odom_rpe)
        print(str(mydate),cond,': The proposed method achieves a reduction of % in median RPE with respect to corrected odometry ',reduction)

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
















