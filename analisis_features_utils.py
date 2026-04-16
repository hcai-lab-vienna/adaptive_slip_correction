import pandas as pd
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from PH import PageHinkley
import os
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from evo.core import  sync

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

fechas_train = {"2025-01-10","2025-03-10", "2025-04-15","2025-09-24","2024-11-21","2025-08-20", "2025-10-14","2024-11-28"}#"2025-06-26",
fechas_test = {"2025-01-30", "2025-05-28","2025-11-03"}
fecha_SNOW_SNOWING_test = {"2025-03-10"}
DRIVETRAIN_DATE=[("2025-01-10",1),("2025-03-10",1), ("2025-04-15",0),("2025-06-26",0),("2025-09-24",0),("2025-11-03",0),("2024-11-21",0),("2025-01-30",1),("2025-05-28",0),("2025-08-20",0),("2025-10-14",0),("2024-11-28",0)]
CONDITIONS_DATE=[("2025-01-10",1),("2025-03-10",2), ("2025-04-15",4),("2025-06-26",3),("2025-09-24",3),("2025-11-03",4),("2024-11-21",3),("2025-01-30",1),("2025-05-28",3),("2025-08-20",3),("2025-10-14",3),("2024-11-28",1)]
CONDITION_CLASSIFICATION_EXPLAINATION=[(1,'snow on the road, not snowing'),
                          (2,'snow on the road, snowing'),
                          (3,'clear road, not raining'),
                          (4,'clear road, raining')]
CONDITION_CLASSIFICATION=[(1,'snow_road'),
                          (2,'snow_snowing'),
                          (3,'clear_road'),
                          (4,'clear_raining')]
DRIVETRAIN_CLASSIFICATION=[(0,'Wheel'),(1,'Track')]

'''FEATURES_METEO = [
    'Rain_accumulation',
    #'T_probe_Avg',
    'RH',#%
    #'T_DP',#C
    #'CS106_Corrected_mbar',
    'SnowDepth_Avg',
    #'Drivetrain_type'
]
FEATURES_SV1 = [
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
    'ang_vel_odom_z',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z'
]
FEATURES_SV2 = [
    'NORMA_lin_acc_imu',
    'NORMA_ang_vel_imu',
    'grav_x',
    'grav_y',
    'grav_z',
    'lin_vel_twist',
    'ang_vel_twist',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z'
]
LABEL_SV='SV'
numeric_limits_SV=[-5,0,5]
FEATURES_L_W_1 = [
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
    'ang_vel_odom_z',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z'
]
FEATURES_L_W_2 = [
    'NORMA_lin_acc_imu',
    'NORMA_ang_vel_imu',
    'grav_x',
    'grav_y',
    'grav_z',
    'lin_vel_twist',
    'ang_vel_twist',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z'
]
lABEL_L_W='TARGET'# GT LINEAR VELOCITY
'''

def get_season(fecha_str):
    fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    year = fecha.year

    # Definir límites
    spring_start = datetime(year, 3, 20)
    summer_start = datetime(year, 6, 21)
    autumn_start = datetime(year, 9, 23)
    winter_start = datetime(year, 12, 21)

    if fecha >= winter_start or fecha < spring_start:
        return "winter"
    elif fecha >= spring_start and fecha < summer_start:
        return "spring"
    elif fecha >= summer_start and fecha < autumn_start:
        return "summer"
    else:
        return "autumn"

# Function to create the DataFrame with only the index
def crear_dataframe_con_indices(indices):
    # Crear un DataFrame vacío con el índice proporcionado
    df = pd.DataFrame(index=indices)
    return df

def rellenar_soil_Condition(df):
    # make sure datetime in index
    df.index = pd.to_datetime(df.index)
    #  array to dic
    condition_dict = {
        pd.to_datetime(date): value
        for date, value in CONDITIONS_DATE
    }
    drivetrain_dict = {
        pd.to_datetime(date): value
        for date, value in DRIVETRAIN_DATE
    }
    # create the column
    df["Soil_type"] = pd.Series(df.index.date,index=df.index).map(condition_dict)
    df["Drivetrain_type"] = pd.Series(df.index.date,index=df.index).map(drivetrain_dict)
    return df

# -------------------------------------------------
# Function to evaluate  T-kan SGDRegressor PassiveAgressiveRegressor
# -------------------------------------------------
def evaluate_modelPH(name, model,X_train, y_train,X_test,y_test, flag,online=False):
    print(f"\n===== {name} =====")

    '''flag=0 SGDRegressor
    flag = 1    SGDPassive Agressive
    flag = 2    SGDRegressor'''

    tracemalloc.start()
    start_train = time.time()

    # Entrenamiento inicial
    if((online==True) & (name!='T-KAN (real)')):
        # Si modelo online, usar partial_fit con train
        model.partial_fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_train
    start_test = time.time()

    preds = []

    #if name=='T-KAN (real)':
    # Inicializamos Page-Hinkley
    if flag==0:
        ph = PageHinkley(delta=0.0019, lambda_=26,alpha=0.999)  # ajustar 
    elif flag==1:
        ph = PageHinkley(delta=0.002, lambda_=20,alpha=0.999)  # ajustar
    else:
        ph = PageHinkley(delta=0.1, lambda_=150, alpha=0.95)  # ajustar


    drifts = []
    cooldown = 100  # mínimo número de pasos entre drifts consecutivos
    last_drift = -cooldown

    buffer_X = []
    buffer_y = []
    window = 20

    for i in range(len(X_test)):
        x = X_test[i].reshape(1, -1)
        y_true = y_test[i]

        # Predicción
        if name=='T-KAN (real)':
            y_pred = float(model.predict(x))
            preds.append(y_pred)
        else:
            y_pred = float((model.predict(x))[0])
            preds.append(y_pred)

        # Actualización modelo si es online
        if online:
            #model.partial_fit(x, [y_true])
            buffer_X.append(x)
            buffer_y.append(y_true)

            if len(buffer_X) > window:
                buffer_X.pop(0)
                buffer_y.pop(0)
            model.partial_fit(np.vstack(buffer_X), buffer_y)

        # Calculamos error actual
        error = y_true - y_pred  # Mantener signo para PH más sensible
        drift = ph.update(error)

        # Detectamos solo drifts separados por cooldown
        if drift and (i - last_drift) >= cooldown:
            drifts.append(i)
            last_drift = i

    test_time = time.time() - start_test
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Métricas
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Imprimir resultados
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
    print("Train time:", train_time)
    print("Test time:", test_time)
    print("Peak memory (KB):", peak / 1024)
    print("Drifts detectados:", drifts)

    # Gráfica
    plt.figure(figsize=(12, 5))
    plt.title(f"{name}")
    plt.scatter(np.arange(len(y_test)), y_test, label="Real", color='r', linewidth=1,s=1)
    plt.scatter(np.arange(len(y_test)), preds, label="Prediction", color='b',linewidth=1,s=1)

    # Marcar drifts
    '''for d in drifts:
        plt.axvline(x=d, color='black', linestyle='--', alpha=0.7)
    '''
    plt.legend()
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.show(block=True)

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
        X_seq.append(X[i - lags:i])  # ventana completa
        y_seq.append(y[i])           # valor siguiente

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # aplanar para MLPRegressor
    X_seq = X_seq.reshape(X_seq.shape[0], -1)

    return X_seq, y_seq
def create_sequences(X, y, lags=15):
    X_seq = []
    y_seq = []

    for i in range(lags, len(X)):
        X_seq.append(X[i - lags:i])  # ventana completa
        y_seq.append(y[i])  # valor siguiente

    return np.array(X_seq), np.array(y_seq)
def columns_all_nan(df):
    return df.columns[df.isna().all()].tolist()
def fix_decimal(x, n_integer_digits=10):
    x = float(x)
    digits = int(np.floor(np.log10(abs(x)))) + 1
    shift = digits - n_integer_digits
    return x / (10 ** shift)

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

        # Mantener solo la primera aparición de cada timestamp
        meteo = meteo.drop_duplicates()
        # Ordenar por tiempo
        meteo = meteo.sort_values("TIMESTAMP")
        # Usar como índice
        meteo = meteo.set_index("TIMESTAMP")
        #meteo["TIMESTAMP"] = meteo["TIMESTAMP"].apply(fix_decimal)
        meteo_all_zero = meteo.select_dtypes(include="number").columns[
            (meteo.select_dtypes(include="number") == 0).all()
        ]
        #print(meteo_all_zero.tolist())
        meteo = meteo.loc[(meteo != 0).any(axis=1)]
        METEO_COLUMNS=['Rain_mm_Tot', 'Rain_accumulation','T_probe_Avg','RH_probe','T_DP_Probe','CS106_Corrected_mbar']
        METEO_COLUMNS=['Rain_accumulation','T_probe_Avg','RH_probe','T_DP_Probe','CS106_Corrected_mbar']
        meteo=meteo[METEO_COLUMNS]
        meteo = meteo.rename(columns={'RH_probe': 'RH', 'Rain_accumulation': 'Rain_accum'})
        meteo = meteo.apply(pd.to_numeric, errors="coerce")
    else:
        print(f"⚠️ Archivo vacío o no válido: {file_meteo}")
        file_meteo = f"{path}\meteo_data.dat"

        meteo = pd.read_csv(
            file_meteo,
            header=1,
            skiprows=[2, 3],
            parse_dates=["TIMESTAMP"]
        )
        meteo.drop('RECORD', axis=1, inplace=True)
        meteo = meteo.drop_duplicates()
        # Usar como índice
        meteo = meteo.set_index("TIMESTAMP")

        #print('NAN', (meteo.isna().mean() * 100).sort_values(ascending=False))

        meteo_all_zero = meteo.select_dtypes(include="number").columns[
            (meteo.select_dtypes(include="number") == 0).all()
        ]
        #print(meteo_all_zero.tolist())

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
        # Convertimos a float
        snow["TIMESTAMP"]=snow["TIMESTAMP"].astype(float)
        #snow["TIMESTAMP"] = snow["TIMESTAMP"].apply(fix_decimal)
        snow = snow.drop_duplicates()
        # Ordenar por tiempo
        snow = snow.sort_values("TIMESTAMP")
        # Usar como índice
        snow = snow.set_index("TIMESTAMP")
        #print('NAN',(snow.isna().mean() * 100).sort_values(ascending=False))
        snow_all_zero = snow.select_dtypes(include="number").columns[
            (snow.select_dtypes(include="number") == 0).all()
        ]
        #print(snow_all_zero.tolist())
        cols = [c for c in snow.columns if str(c).startswith("SDMS40_Distance_Points")]
        snow["SDMS40_Distance_Avg"] = snow[cols].mean(axis=1)
        SNOW_COLUMNS = ['SDMS40_Depth_Avg', 'SDMS40_Board_Temperature',
           'SDMS40_Heater_Low_Threshold_Temperature', 'SDMS40_Laser_Temperature','SDMS40_Distance_Avg']
        SNOW_COLUMNS = ['SDMS40_Depth_Avg']
        snow = snow[SNOW_COLUMNS]
        snow = snow.rename(columns={'SDMS40_Depth_Avg': 'SnowDepth_Avg'})
    else:
        print(f"⚠️ Archivo vacío o no válido: {file_path}")
        file_path = f"{path}\snow_data.dat"
        snow = pd.read_csv(
            file_path,
            header=1,
            skiprows=[2, 3],
            parse_dates=["TIMESTAMP"]
        )
        snow.drop('RECORD', axis=1, inplace=True)
        snow = snow.drop_duplicates()
        # Usar como índice
        snow = snow.set_index("TIMESTAMP")
        #print('NAN', (snow.isna().mean() * 100).sort_values(ascending=False))
        snow_all_zero = snow.select_dtypes(include="number").columns[
            (snow.select_dtypes(include="number") == 0).all()
        ]
        #print(snow_all_zero.tolist())
        cols = [c for c in snow.columns if str(c).startswith("SDMS40_Distance_Points")]
        snow["SDMS40_Distance_Avg"] = snow[cols].mean(axis=1)
        SNOW_COLUMNS = ['SDMS40_Depth_Avg', 'SDMS40_Board_Temperature',
                        'SDMS40_Heater_Low_Threshold_Temperature', 'SDMS40_Laser_Temperature',
                        'SDMS40_Distance_Avg']
        SNOW_COLUMNS = ['SDMS40_Depth_Avg']
        snow = snow[SNOW_COLUMNS]
        snow = snow.rename(columns={'SDMS40_Depth_Avg': 'SnowDepth_Avg'})

    #Calculamos el indice comun
    dftmp = [meteo, snow]
    dfs_validostmp = []
    for df in dftmp:
        if df is not None and not df.empty:
            dfs_validostmp.append(df)
        else:
            print(" DataFrame vacío descartado")
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
            #print(interseccion)

            start = max(df.index.min() for df in dfs_validostmp)
            end = min(df.index.max() for df in dfs_validostmp)

            dfs_validostmp = [df.loc[start:end] for df in dfs_validostmp]
        tmp_index = dfs_validostmp[0].index

        # ---------- TERRAIN ----------
        # Crear el DataFrame con solo los índices
        df_terreno = crear_dataframe_con_indices(tmp_index)

        # Rellenar las columnas específicas
        df_terreno = rellenar_soil_Condition(df_terreno)

        name_file=str('Terrain_')+str(season)+str('.csv')
        df_terreno.to_csv(f"{path}/{name_file}",sep=',', header=True,index=True)

        # salvar el DataFrame final
        #print(df_terreno)
    else:
        print(f"⚠️ DATAFRAME TERAIN vacío.")
        df_terreno = pd.DataFrame()

    # ---------- MERGE ----------
    dfmeteo = [meteo, snow, df_terreno]
    dfs_validos = []
    for df in dfmeteo:
        if df is not None and not df.empty:
            dfs_validos.append(df)
        else:
            print(" DataFrame vacío descartado")

    if len(dfs_validostmp) != 0:
        if flag_INTERSECTION!=0:
            first_valid_value = snow['SnowDepth_Avg'].iloc[0]
            master_index = dfs_validostmp[0].index  # odom NEED TO BE ONE AT THE END. ( THe SAME)
            aligned = []
            for df in dfmeteo:
                df_interp = (
                    df.reindex(master_index)
                )
                aligned.append(df_interp)

            df_meteo_final = pd.concat(aligned, axis=1)
            df_meteo_final['SnowDepth_Avg']=first_valid_value
        else:
            master_index = dfs_validostmp[0].index  # odom NEED TO BE ONE AT THE END. ( THe SAME)
            aligned = []
            for df in dfmeteo:
                df_interp = (
                    df.reindex(master_index)
                )
                aligned.append(df_interp)
            df_meteo_final = pd.concat(aligned, axis=1)
    else:
        print(f"⚠️ DATAFRAME FINAL vacío.")
        df_meteo_final = pd.DataFrame()

    return df_meteo_final
def load_trajectory_data2(fecha,traj):
    # Load data
    trajectory_dir = get_trajectory_dir(deployment=fecha, trajectory=traj)
    # Transforms
    transform_manager = get_transforms(trajectory_dir)
    # IMUs
    imu_name = 'vectornav'
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
    vel_gt = np.linalg.norm(np.array(p_rel_gt)[:,:3,3], axis=1) / delta_ts_gt
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
    odom_dt_index =pd.to_datetime(traj_odom_imu.timestamps[1:] * 1e9)
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

    '''p_rel_gt = relative_pose_from_trajectories([traj_gt])[0]
    delta_ts_gt = traj_gt.timestamps[1:] - traj_gt.timestamps[:-1]
    vel_gt_COMP = [velocities_from_deltaT(dT, dt) for dT, dt in zip(p_rel_gt, delta_ts_gt)]
    lin_vel_gt = [vl for vl, va in vel_gt_COMP]
    ang_vel_gt = [va for vl, va in vel_gt_COMP]'''

    DATASET = pd.DataFrame(
        np.hstack([
            accel_grav_compensated_sync,
            gyro_sync,
            g_body_sync,
            lin_vel_twist[2:], ang_vel_twist[2:],
            lin_vel_cmd,
            ang_vel_cmd,
            gt_df['vel_gt'].values.reshape(-1, 1)
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
            'TARGET'
        ]
    )

    if len(odom_dt_index)!=DATASET.shape[0]:
        print('DIMMENSIONAL ERROR')

    DATASET = DATASET.set_index(odom_dt_index)
    DATASET = DATASET.dropna()

    #plot_trajectories(traj_odom_imu, traj_odom, traj_gt)

    return DATASET
def plot_trajectories(traj_odom_imu,traj_odom,traj_gt):
    plt.figure(figsize=(12, 5))
    plt.title("PLOT trajectories")

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
def analyze_index(df):
    diffs = df.index.to_series().diff().dropna()
    return {
        "freq_mean": diffs.mean(),
        "freq_median": diffs.median(),
        "freq_min": diffs.min(),
        "freq_max": diffs.max(),
        "gaps_over_1s": (diffs > pd.Timedelta("1s")).sum(),
        "max_gap": diffs.max(),
        "count": len(df)
    }

def FEATURE_IMP(input,flag_trin_test, features,taget_lab):
    print("\nTraining independent models independientes...")

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
    print("\nImportancia de variables:")
    print(fi)
    if flag_trin_test==0:
        name=' TRAIN'
    else:
        name = ' TEST'

    title_plt=[ "Changes in the relationship between SV "+str(name) ]

    fi.plot(kind="bar", title=title_plt[0],color=["gold", "blue","green","orange"],fontsize=18)
    plt.ylabel("Feature importance",fontsize=16)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=9)
    plt.show(block=True)

def tratamiento_XGB_season(input,test,features,LABEL_SV):
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
    y_test_snow_road_index=df_TEST_snow_road.index
    y_TEST_snow_road = df_TEST_snow_road[LABEL_SV].values
    X_TEST_clear_road = df_TEST_clear_road[features].values
    y_test_clear_road_index = df_TEST_clear_road.index
    y_TEST_clear_road = df_TEST_clear_road[LABEL_SV].values
    X_TEST_clear_raining = df_TEST_clear_raining[features].values
    y_test_clear_raining_index = df_TEST_clear_raining.index
    y_TEST_clear_raining = df_TEST_clear_raining[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_snow_road, X_clear_road, X_clear_raining]
    LABELS_ENTRENAMIENTO = [y_snow_road, y_clear_road, y_clear_raining]
    INDICES = [y_test_snow_road_index, y_test_clear_road_index, y_test_clear_raining_index,]
    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]

    results = []
    prediction = []
    condiciones = ["snow_road", "clear_road", "clear_raining"]
    for est1, X_train, y_train, X_test, y_test in zip(
            condiciones,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Entrenando condiciones: {est1}")

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
                condiciones,
                DATOS_TEST,
                LABELS_TEST
        ):
            print(f"Predicion condiciones: {est2}")
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

    print("Matriz MAE:")
    print(matriz_mae)

    print("\nMatriz R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("XGB MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("XGB R2 (train vs test)")
    plt.show()

def tratamiento_SGDseason(input,test,features,LABEL_SV):
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
    INDICES = [y_test_snow_road_index,  y_test_clear_road_index, y_test_clear_raining_index, ]
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
        print(f"Entrenando condiciones: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        lag = len(features)

        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

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

            print(f"Predicion condiciones: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            # SGD
            sgd1 = SGDRegressor(
                max_iter=1,
                warm_start=True,
                loss='epsilon_insensitive',  # Passive Aggressive Esto actúa como SVR online.
                learning_rate='pa1',
                eta0=1,
                alpha=0.001,
                penalty=None
            )
            title_plt=f"Real {est1} vs Prediction {est2}"+ " SGDRegressor without PartialFit"
            resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH(title_plt, sgd1,
                                                                              X_train, y_train, X_test, y_test, 0,
                                                                              online=False)
            prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
                np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
            prediction.append(prediccion_sgd_offlinenversa)

            y_test_inverse = y_scaler.inverse_transform(
                np.array(y_test).reshape(-1, 1)).flatten()
            mae_pred = mean_absolute_error(y_test_inverse, prediccion_sgd_offlinenversa)
            r2 = r2_score(y_test_inverse, prediccion_sgd_offlinenversa)

            coefs = np.abs(sgd1.coef_)  # shape (32,)
            # reshape → (n_features, n_lags)
            coefs_reshaped = coefs.reshape(len(features), 2)
            importance_per_feature = coefs_reshaped.sum(axis=1)

            df_sgd = pd.DataFrame({
                "feature": features,
                "importance": importance_per_feature
            }).sort_values(by="importance", ascending=False)

            values = df_sgd["importance"]
            colors = cm.viridis((values - values.min()) / (values.max() - values.min()))

            plt.figure(figsize=(8, 5))
            plt.barh(df_sgd["feature"], df_sgd["importance"],color=colors)
            plt.gca().invert_yaxis()
            plt.title(str(est1)+" Feature Importance - SGD")
            plt.show()

            '''plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(lag, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter( np.arange(lag, len(y_test)),prediccion_sgd_offlinenversa, label="Prediction SGD", color='turquoise', linewidth=1, s=1)
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
                'prediccion': prediccion_sgd_offlinenversa
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

    print("Matriz MAE:")
    print(matriz_mae)

    print("\nMatriz R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("SGD MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("SGD R2 (train vs test)")
    plt.show()

def tratamiento_SGDPARTIALseason(input,test,features,LABEL_SV):
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
    INDICES = [y_test_snow_road_index,  y_test_clear_road_index, y_test_clear_raining_index, ]
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
        print(f"Entrenando condiciones: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        lag = len(features)

        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

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

            print(f"Predicion condiciones: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            sgd2 = SGDRegressor(
                max_iter=1,
                warm_start=True,
                loss='epsilon_insensitive',
                learning_rate='pa1',
                eta0=1,
                alpha=0.001,
                penalty=None
            )
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

            coefs = np.abs(sgd2.coef_)  # shape (32,)
            # reshape → (n_features, n_lags)
            coefs_reshaped = coefs.reshape(len(features), 2)
            importance_per_feature = coefs_reshaped.sum(axis=1)

            df_sgd = pd.DataFrame({
                "feature": features,
                "importance": importance_per_feature
            }).sort_values(by="importance", ascending=False)

            values = df_sgd["importance"]
            colors = cm.viridis((values - values.min()) / (values.max() - values.min()))

            plt.figure(figsize=(8, 5))
            plt.barh(df_sgd["feature"], df_sgd["importance"],color=colors)
            plt.gca().invert_yaxis()
            plt.title(str(est1) + " Feature Importance - SGD PARTIAL")
            plt.show()

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

    print("Matriz MAE:")
    print(matriz_mae)

    print("\nMatriz R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("SGD PARTIAL FIT MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title(" SGD PARTIAL FIT R2 (train vs test)")
    plt.show()

def tratamiento_TKANseason(input,test,features,LABEL_SV):
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
    INDICES = [y_test_snow_road_index, y_test_clear_road_index, y_test_clear_raining_index, ]
    DATOS_TEST = [X_TEST_snow_road, X_TEST_clear_road, X_TEST_clear_raining]
    LABELS_TEST = [y_TEST_snow_road, y_TEST_clear_road, y_TEST_clear_raining]

    results = []
    prediction = []
    condiciones = ["snow_road", "snow_snowing", "clear_road", "clear_raining"]
    for est1, X_train, y_train, X_test, y_test in zip(
            condiciones,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Entrenando condiciones: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        lag = len(features)
        XTRAIN_lag, yTRAIN_lag = create_sequences_flat(X_train, y_train, lags=lag)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag

        contador=0
        for est2, X_test, y_test in zip(
                condiciones,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_sequences_flat(X_test, y_test, lags=lag)  # create_lags
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Predicion condiciones: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            # T-KAN (aproximated with  MLP)
            tkan1 = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='identity',
                max_iter=8000,  #
                early_stopping=False,
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

    print("Matriz MAE:")
    print(matriz_mae)

    print("\nMatriz R2:")
    print(matriz_r2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_mae, annot=True, cmap="coolwarm")
    plt.title("TKAN MAE (train vs test)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_r2, annot=True, cmap="coolwarm")
    plt.title("TKAN R2 (train vs test)")
    plt.show()

def tratamiento_XGB_WHOLE(X,y,X_t,y_t,features):
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
def tratamiento_SGD_WHOLE(X,y,X_t,y_t,features,label_target):
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

    XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=lag)  # create_lags
    X_train, y_train = XTRAIN_lag, yTRAIN_lag
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=lag)  # create_lags
    X_test, y_test = XTEST_lag, yTEST_lag
    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # SGD
    sgd1 = SGDRegressor(
        max_iter=1,
        warm_start=True,
        loss='epsilon_insensitive',  # Passive Aggressive Esto actúa como SVR online.
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        penalty=None
    )
    resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH("SGDRegressor without PartialFit", sgd1,
                                                                      X_train, y_train, X_test, y_test, 0,
                                                                      online=False)
    prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
        np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
    prediction.append(prediccion_sgd_offlinenversa)
    mae_fit = resultados_sgd_offline['mae']
    r2fit = resultados_sgd_offline['r2']

    sgd2 = SGDRegressor(
        max_iter=1,
        warm_start=True,
        loss='epsilon_insensitive',
        learning_rate='pa1',
        eta0=1,
        alpha=0.001,
        penalty=None
    )
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
        # Añadir texto en la figura (x, y son coordenadas normalizadas 0-1)
        plt.text(
            1.05, 0.1,  # posición a la derecha del plot
            f"SGD MAE = {mae_fit:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.2,  # posición a la derecha del plot
            f"SGD R2 = {r2fit:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.3,  # posición a la derecha del plot
            f"SGD Partial MAE = {mae_partial:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.4,  # posición a la derecha del plot
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
def tratamiento_TKAN_WHOLE(X,y,X_t,y_t,features):
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
        n_iter_no_change=50,
        validation_fraction=0.0005,
        random_state=42
    )
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
        cont = 0
        for d in drift1:
            if cont == 0:
                plt.axvline(x=d, color='brown', linestyle='--', alpha=0.7, label=' TKAN Drift')
            else:
                plt.axvline(x=d, color='brown', linestyle='--', alpha=0.7)
            cont += 1
        # Añadir texto en la figura (x, y son coordenadas normalizadas 0-1)
        plt.text(
            1.05, 0.1,  # posición a la derecha del plot
            f"SGD MAE = {mae_tkan:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.text(
            1.05, 0.2,  # posición a la derecha del plot
            f"SGD R2 = {r2tkan:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center'
        )
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel('SV')
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
def VISUALIZACION_MEJORA(X_t,y_t,pred,features,lab_target,title):
    indice = y_t.index

    lag = len(features)
    pred = pd.Series(pred, index=indice[lag:])

    fechas_unicas = y_t.index.normalize().unique()
    for fecha in fechas_unicas:
        mask = (pd.to_datetime(indice[lag:]).normalize() == fecha)

        horas = indice[lag:][mask]
        reales = y_t[lag:][mask]
        odom_lin_vel=X_t[lag:][mask]['lin_vel_odom_x']
        odom_ang_vel = X_t[lag:][mask]['ang_vel_odom_z']
        pred = pred[mask]
        print(odom_ang_vel.shape[0],reales.shape[0],odom_lin_vel.shape[0],pred.shape[0])

        delta_ts = (horas[1:] - horas[:-1]) / np.timedelta64(1, 's')
        print(delta_ts.shape[0])

        p_rel_gt_rec, p_gt_rec = integrate_body_twists(reales[1:], odom_ang_vel[1:], delta_ts)
        p_rel_odom_rec, p_odom_rec = integrate_body_twists(odom_lin_vel[1:], odom_ang_vel[1:], delta_ts)
        p_rel_pred_rec, p_pred_rec = integrate_body_twists(pred[1:], odom_ang_vel[1:], delta_ts)
        gtcoor_x = [M[0, 3] for M in p_rel_gt_rec]
        gtcoor_y = [M[1, 1] for M in p_rel_gt_rec]
        odomcoor_x = [M[0, 3] for M in p_rel_odom_rec]
        odomcoor_y = [M[1, 1] for M in p_rel_odom_rec]
        predcoor_x = [M[0, 3] for M in p_rel_pred_rec]
        predcoor_y = [M[1, 1] for M in p_rel_pred_rec]
        fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharey=False)
        titulo = str(title+ str (fecha))
        fig.suptitle(titulo, fontsize=16)
        plt.title(f"Results for the day {fecha.date()}")
        plt.scatter(gtcoor_x,gtcoor_y, label="Real", color='r', linewidth=1, s=1)
        plt.scatter(predcoor_x,predcoor_y, label="Prediction", color='blue', linewidth=1, s=1)
        plt.scatter(odomcoor_x, odomcoor_y, label="Odometry corrected", color='g', linewidth=1, s=1)
        plt.legend()
        plt.xlabel("Coord X")
        plt.ylabel('Coord Y')
        plt.show(block=True)
    print('FIN')








