import pandas as pd

pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from PH import PageHinkley
import os
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from evo.core import  sync
from imu_utils import estimate_gravity
from fomo_utils import get_odom_trajectory, get_gt_trajectory
from trajectory_utils import (kabsch_algorithm,compute_rpe_from_rel_pose,
                                                       integrate_body_twists,rmse,orientations_from_positions,
                                                       velocities_from_trajectories,reduce_to_ids)
import copy

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "..\\..\\..\\OSR\\code\\fomo-dataset"
NEW_FOMO_PATH = "..\\..\\..\\OSR\\code\\New_fomo_DATASET\\"

fechas_train = {"2025-01-10","2025-03-10", "2025-04-15","2025-06-26","2025-09-24", "2025-11-03","2024-11-21"}
fechas_test = {"2025-01-29", "2025-05-28","2025-08-20", "2025-10-14","2024-11-28"}
DRIVETRAIN_DATE=[("2025-01-10",1),("2025-03-10",1), ("2025-04-15",0),("2025-06-26",0),("2025-09-24",0),("2025-11-03",0),("2024-11-21",0),("2025-01-29",1),("2025-05-28",0),("2025-08-20",0),("2025-10-14",0),("2024-11-28",0)]
CONDITIONS_DATE=[("2025-01-10",6),("2025-03-10",3), ("2025-04-15",5),("2025-06-26",6),("2025-09-24",6),("2025-11-03",2),("2024-11-21",6),("2025-01-29",1),("2025-05-28",6),("2025-08-20",6),("2025-10-14",6),("2024-11-28",0)]
CONDITION_CLASSIFICATION=[(0,'little snow on the road'),
                          (1,'snow on the road'),
                          (2,'little snow snowing'),
                          (3,'snow snowing'),
                          (4,'rain road/Gravel'),
                          (5,'snow road/Gravel'),
                          (6,'clear road/Gravel')]
DRIVETRAIN_CLASSIFICATION=[(0,'Wheel'),(1,'Track')]

features_terrain=['Soil Condition']  #'Slope'== INFO GRAVITY

ARRAY_Δx=[]
ARRAY_Δy=[]
ARRAY_date=[]

FEATURES_METEO = [
    "lin_vel",
    "odom_acc_norm",
    "imu1_acc_norm",
    "imu1_gyro_norm",
    "imu2_acc_norm",
    'Soil Condition'
]

FEATURES_SV = [
    "lin_vel",
    "odom_acc_norm",
    "imu1_acc_norm",
    "imu1_gyro_norm",
    "imu2_acc_norm",
    "imu2_gyro_norm"
]

LABEL_SV='SV'
numeric_limits_SV=[]

FEATURES_L_W = [
    "lin_vel",
    "odom_acc_norm",
    "imu1_acc_norm",
    "imu1_gyro_norm",
    "imu2_acc_norm",
    "imu2_gyro_norm",
    "SV",
    'LIN_VEL_x',
    'LIN_VEL_y',
    'LIN_VEL_z',
    'ANG_VEL_x',
    'ANG_VEL_y',
    'ANG_VEL_z'
]
LABEL_L_W=[
    'GT_LIN_VEL_x',
    'GT_LIN_VEL_y',
    'GT_LIN_VEL_z',
    'GT_ANG_VEL_x',
    'GT_ANG_VEL_y',
    'GT_ANG_VEL_z'
]


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
    plt.title(f"{name}  : Real vs Prediction")
    plt.scatter(np.arange(len(y_test)), y_test, label="Real", color='r', linewidth=1,s=1)
    plt.scatter(np.arange(len(y_test)), preds, label="Prediction", color='b',linewidth=1,s=1)

    # Marcar drifts
    for d in drifts:
        plt.axvline(x=d, color='black', linestyle='--', alpha=0.7)

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

    # 🔥 aplanar para MLPRegressor
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


def From_GNSS_2_ODOM(coordgnssX,coordgnssY):
    return coordgnssX-Δx,coordgnssY-Δy


def From_ODOM_2_GNSS(coord_odomX,coord_odomY):
    return coord_odomX+Δx,coord_odomY+Δy


def trasladar_serie(serie, cantidad):
    return serie + cantidad


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
        meteo["TIMESTAMP"] = meteo["TIMESTAMP"].apply(fix_decimal)
        # Mantener solo la primera aparición de cada timestamp
        meteo = meteo.drop_duplicates()
        # Ordenar por tiempo
        meteo = meteo.sort_values("TIMESTAMP")
        # Usar como índice
        meteo = meteo.set_index("TIMESTAMP")
        meteo_all_zero = meteo.select_dtypes(include="number").columns[
            (meteo.select_dtypes(include="number") == 0).all()
        ]
        print(meteo_all_zero.tolist())
        meteo = meteo.loc[(meteo != 0).any(axis=1)]
        METEO_COLUMNS=['Rain_mm_Tot', 'Rain_accumulation','T_probe_Avg','RH_probe','T_DP_Probe','CS106_Corrected_mbar']
        METEO_COLUMNS=['Rain_accumulation','T_probe_Avg','RH_probe','T_DP_Probe','CS106_Corrected_mbar']
        meteo=meteo[METEO_COLUMNS]
        meteo = meteo.apply(pd.to_numeric, errors="coerce")
    else:
        print(f"⚠️ Archivo vacío o no válido: {file_meteo}")
        meteo = pd.DataFrame()

    # ---------- snow ----------
    file_path=f"{path}\snow_data.csv"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        snow = pd.read_csv(file_path )
        snow.drop(0, axis=0, inplace=True)  # units
        snow.drop(1, axis=0, inplace=True) #NAN
        snow.drop('RECORD', axis=1, inplace=True)
        snow = snow.apply(pd.to_numeric, errors="coerce")
        # Convertimos a float
        snow["TIMESTAMP"] = snow["TIMESTAMP"].apply(fix_decimal)
        snow = snow.drop_duplicates()
        # Ordenar por tiempo
        snow = snow.sort_values("TIMESTAMP")
        # Usar como índice
        snow = snow.set_index("TIMESTAMP")

        print('NAN',(snow.isna().mean() * 100).sort_values(ascending=False))

        snow_all_zero = snow.select_dtypes(include="number").columns[
            (snow.select_dtypes(include="number") == 0).all()
        ]
        print(snow_all_zero.tolist())

        cols = [c for c in snow.columns if str(c).startswith("SDMS40_Distance_Points")]
        snow["SDMS40_Distance_Avg"] = snow[cols].mean(axis=1)
        SNOW_COLUMNS = ['SDMS40_Depth_Avg', 'SDMS40_Board_Temperature',
           'SDMS40_Heater_Low_Threshold_Temperature', 'SDMS40_Laser_Temperature','SDMS40_Distance_Avg']
        SNOW_COLUMNS = ['SDMS40_Depth_Avg']
        snow = snow[SNOW_COLUMNS]
    else:
        print(f"⚠️ Archivo vacío o no válido: {file_path}")
        snow = pd.DataFrame()

    #Calculamos el indice comun
    dftmp = [meteo, snow]
    dfs_validostmp = []
    for df in dftmp:
        if df is not None and not df.empty:
            dfs_validostmp.append(df)
        else:
            print(" DataFrame vacío descartado")
    if len(dfs_validostmp) != 0:
        for df in dfs_validostmp:
            df.index = pd.to_datetime(df.index, unit='s')
            print(df.index.min(), df.index.max())

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
        print(df_terreno)
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
        for df in dfs_validos:
            df.index = pd.to_datetime(df.index, unit='s')
            print(df.index.min(), df.index.max())

        start = max(df.index.min() for df in dfs_validos)
        end = min(df.index.max() for df in dfs_validos)

        dfs_validos = [df.loc[start:end] for df in dfs_validos]

        master_index = dfs_validos[0].index  # odom NEED TO BE ONE AT THE END. ( THe SAME)
        aligned = []
        for df in dfs_validos:
            df_interp = (
                df.reindex(master_index)
            )
            aligned.append(df_interp)
        df_meteo_final = pd.concat(aligned, axis=1)
        df_meteo_final["season"] = season
    else:
        print(f"⚠️ DATAFRAME FINAL vacío.")
        df_meteo_final = pd.DataFrame()

    return df_meteo_final


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


def load_trajectory_files(path,season):
    # ---------- TRAJECTORY ----------
    # Load data
    traj_gt = get_gt_trajectory(path)
    traj_gt_oriented, gt_headings = orientations_from_positions(traj_gt)
    traj_odom, lin_vel_twist, ang_vel_twist = get_odom_trajectory(path)
    delta_ts_twist = traj_odom.timestamps[1:] - traj_odom.timestamps[:-1]

    # Recover linear and angular velocities from trajectories
    lin_vel_gt, ang_vel_gt, traj_gt_oriented, p_rel_gt, delta_ts = velocities_from_trajectories(traj_gt_oriented)

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

    traj_gt_aligned = copy.deepcopy(traj_gt_oriented)
    num_used_poses, r_a, t_a = kabsch_algorithm(
        np.array(p_gt_rec)[:, :3, 3], traj_gt_aligned.positions_xyz
    )
    pos_gt_aligned = np.dot(r_a, (traj_gt_aligned.positions_xyz + t_a).T).T
    print(np.array(p_gt_rec).shape,pos_gt_aligned.shape)
    #print("GT absolute position reconstruction RMSE", rmse(np.array(p_gt_rec)[1:, :3, 3], pos_gt_aligned).mean())
    #print("Odom absolute position reconstruction RMSE",rmse(np.array(p_odom_rec)[:, :3, 3], np.array(p_twist_rec)[:, :3, 3]).mean())
    # ---------- ODOM ----------
    odom = pd.read_csv(f"{path}/odom.csv")
    while odom.shape[0]!=len(lin_vel_odom)+1:
        print('DIMENSIONS ERROR ' ,odom.shape[0],odom.shape)
        break
    # To float
    odom["t"] = odom["t"].apply(fix_decimal)
    odom["LIN_VEL"] = [[0,0,0], *lin_vel_odom]
    odom["ANG_VEL"] = [[0,0,0], *ang_vel_odom]
    odom[["LIN_VEL_x", "LIN_VEL_y", "LIN_VEL_z"]] = pd.DataFrame(odom["LIN_VEL"].tolist(), index=odom.index)
    odom = odom.drop(columns="LIN_VEL")
    odom[["ANG_VEL_x", "ANG_VEL_y", "ANG_VEL_z"]] = pd.DataFrame(odom["ANG_VEL"].tolist(), index=odom.index)
    odom = odom.drop(columns="ANG_VEL")
    # Keep only the first occurrence of each timestamp
    odom = odom.drop_duplicates(subset=["t"], keep="first")
    # Sort by time
    odom = odom.sort_values("t")
    # Use as index
    odom = odom.set_index("t")

    print ('NAN',(odom.isna().mean() * 100).sort_values(ascending=False))
    protected_cols = ["LIN_VEL_x", "LIN_VEL_y", "LIN_VEL_z","ANG_VEL_x", "ANG_VEL_y", "ANG_VEL_z"]
    odom_all_zero = odom.select_dtypes(include="number").columns[
        (odom.select_dtypes(include="number") == 0).all()
    ]
    print(odom_all_zero.tolist())
    cols_to_drop = [
        col for col in odom_all_zero.tolist()
        if col not in protected_cols
    ]

    odom = odom.drop(columns=cols_to_drop)

    odom = odom.dropna(axis=1)
    kk=odom[(odom == 0).sum(axis=1) > 1]
    odom = odom[(odom != 0).sum(axis=1) > 1]
    odom = odom.add_prefix('omom_')

    # ---------- cmd_VELOCITY ----------
    METEO_PATH = path / "metadata"
    cmd_vel = pd.read_csv(f"{METEO_PATH}/cmd_velocity.csv")
    # To float
    cmd_vel["timestamp"] = cmd_vel["timestamp"].apply(fix_decimal)
    # Keep only the first occurrence of each timestamp
    cmd_vel = cmd_vel.drop_duplicates(subset=["timestamp"], keep="first")
    # Sort by time
    cmd_vel = cmd_vel.sort_values("timestamp")
    # Use as index
    cmd_vel = cmd_vel.set_index("timestamp")

    print('NAN', (cmd_vel.isna().mean() * 100).sort_values(ascending=False))

    cmd_vel_all_zero = cmd_vel.select_dtypes(include="number").columns[
        (cmd_vel.select_dtypes(include="number") == 0).all()
    ]
    print(cmd_vel_all_zero.tolist())

    if cmd_vel.isna().any().any()==True:
        print('AQUI')
        print(cmd_vel.isna().sum())

    cmd_vel= cmd_vel.drop(columns=cmd_vel_all_zero.tolist())
    cmd_vel = cmd_vel.dropna(axis=1)
    cmd_vel = cmd_vel[(cmd_vel != 0).sum(axis=1) > 1]
    cmd_vel = cmd_vel.add_prefix('cmd_vel_')

    # ---------- FIRST IMU (VectorNav) ----------
    imu = pd.read_csv(f"{path}/vectornav.csv")
    # To float
    imu["t"] = imu["t"].apply(fix_decimal)
    # Keep only the first occurrence of each timestamp
    imu = imu.drop_duplicates(subset=["t"], keep="first")
    # Sort by time
    imu = imu.sort_values("t")
    # Use as index
    imu = imu.set_index("t")
    print('NAN',(imu.isna().mean() * 100).sort_values(ascending=False))
    imu_all_zero = imu.select_dtypes(include="number").columns[
        (imu.select_dtypes(include="number") == 0).all()
    ]
    print(imu_all_zero.tolist())
    imu = imu.drop(columns=imu_all_zero.tolist())
    imu = imu.dropna(axis=1)
    imu = imu[(imu != 0).sum(axis=1) > 1]
    tmp=imu['ax']
    imu['ax'] =  imu['ay']
    imu['ay'] = - tmp

    accel = np.array(imu.loc[:, ['ax', 'ay', 'az']])
    gyro = np.array(imu.loc[:, ['wx', 'wy', 'wz']])
    freq = 200.0
    dt = 1 / freq
    g_body = estimate_gravity(accel, gyro, dt, g=9.80665, kp=2.0, ki=0.05)
    imu[["gBODY_X", "gBODY_Y", "gBODY_Z"]] = g_body
    imu = imu.add_prefix('vectorNAv_')

    # ---------- SECOND IMU (VectorNav) ----------
    imu2 = pd.read_csv(f"{path}/xsens.csv")
    # To float
    imu2["t"] = imu2["t"].apply(fix_decimal)
    # Keep only the first occurrence of each timestamp
    imu2 = imu2.drop_duplicates(subset=["t"], keep="first")
    # Sort by time
    imu2 = imu2.sort_values("t")
    # Use as index
    imu2 = imu2.set_index("t")
    print('NAN', (imu2.isna().mean() * 100).sort_values(ascending=False))

    imu2_all_zero = imu2.select_dtypes(include="number").columns[
        (imu2.select_dtypes(include="number") == 0).all()
    ]
    print(imu2_all_zero.tolist())
    imu2 = imu2.drop(columns=imu2_all_zero.tolist())
    imu2 = imu2.dropna(axis=1)
    imu2 = imu2[(imu2 != 0).sum(axis=1) > 1]
    imu2['ax'] = - imu2['ax']
    imu2['ay'] = - imu2['ay']

    accel2 = np.array(imu2.loc[:, ['ax', 'ay', 'az']])
    gyro2 = np.array(imu2.loc[:, ['wx', 'wy', 'wz']])
    freq2 = 200.0
    dt2 = 1 / freq2
    g_body2 = estimate_gravity(accel2, gyro2, dt2, g=9.80665, kp=2.0, ki=0.05)
    imu2[["gBODY_X", "gBODY_Y", "gBODY_Z"]] = g_body2
    imu2 = imu2.add_prefix('xsens_')

    # ---------- GNSS Ground Truth ----------PREGUNTAR BOKU GTCOVARIANCE
    gt = pd.read_csv(
        f"{path}/gt.txt",
        sep=" ",
        names=["t", "gt_x", "gt_y", "gt_z", "qx", "qy", "qz", "qw"]
    )
    # To float
    gt["t"] = gt["t"].astype(float)
    print(lin_vel_gt.shape, gt.shape)
    while gt.shape[0]!=len(lin_vel_gt)+1:
        print('DIMENSIONS ERROR ',gt.shape,len(lin_vel_gt))
        break
    gt["LIN_VEL"] = [[0,0,0], *lin_vel_gt]
    gt["ANG_VEL"] = [[0,0,0], *ang_vel_gt]
    gt[["LIN_VEL_x", "LIN_VEL_y", "LIN_VEL_z"]] = pd.DataFrame(gt["LIN_VEL"].tolist(), index=gt.index)
    gt = gt.drop(columns="LIN_VEL")
    gt[["ANG_VEL_x", "ANG_VEL_y", "ANG_VEL_z"]] = pd.DataFrame(gt["ANG_VEL"].tolist(), index=gt.index)
    gt = gt.drop(columns="ANG_VEL")
    # Keep only the first occurrence of each timestamp
    gt = gt.drop_duplicates(subset=["t"], keep="first")
    # Sort by time
    gt = gt.sort_values("t")
    # Use as index
    gt = gt.set_index("t")
    gtall_zero = gt.select_dtypes(include="number").columns[
        (gt.select_dtypes(include="number") == 0).all()
    ]
    protected_cols = ["LIN_VEL_x", "LIN_VEL_y", "LIN_VEL_z", "ANG_VEL_x", "ANG_VEL_y", "ANG_VEL_z"]
    print(gtall_zero.tolist())
    cols_to_drop = [
        col for col in gtall_zero.tolist()
        if col not in protected_cols
    ]

    gt = gt.drop(columns=cols_to_drop)
    gt = gt.dropna(axis=1)
    gt = gt[(gt != 0).sum(axis=1) > 1]

    # ---------- MERGE ----------
    dfs = [gt, imu, imu2, odom, cmd_vel]
    names = ["gt", "imu", "imu2", "odom", "cmd_vel"]

    # Ensure that all indexes are datetime64[ns]
    for df in dfs:
        df.index = pd.to_datetime(df.index, unit='s',errors="coerce")
        print(df.index.min() , df.index.max() )
        print(df.index.freq)

    # 1) Remove duplicates
    dfs = [df[~df.index.duplicated(keep="first")].sort_index() for df in dfs]

    # 2) Find COMMON time range
    start = max(df.index.min() for df in dfs)
    end = min(df.index.max() for df in dfs)

    print("Actual Sync Window:")
    print(start, "-->", end)

    master_index = dfs[3].index  # odom
    aligned = []
    for df in dfs:
        df = df.sort_index()
        # Create combined index
        new_index = df.index.union(master_index)
        df_interp = (
            df
            .reindex(new_index)
            .interpolate(method="time", limit_direction="both")
            .reindex(master_index)
        )
        aligned.append(df_interp)

    df_final = pd.concat(aligned, axis=1)
    df_final.index = df_final.index.round("99ms")

    if df_final.isna().sum().sum()>0:
        print("NaN total:", df_final.isna().sum().sum())

    first_odom_time = odom.index[0]
    print(first_odom_time)
    # Diferencia absoluta entre cada índice de gt y el timestamp de odom
    time_diffs = abs(gt.index - first_odom_time)

    # GT row with the minimum difference
    closest_gt_idx = time_diffs.argmin()  # position
    closest_gt_time = gt.index[closest_gt_idx]

    print(f"Fila de gt más cercana: {closest_gt_idx}, timestamp: {closest_gt_time}")

    gt_row = gt.iloc[closest_gt_idx]
    odom_row = odom.iloc[0]
    print("Valores GT:", gt_row)
    print("Valores Odom:", odom_row)

    global ARRAY_Δx,ARRAY_Δy,Δx,Δy

    gnss_inicial=[gt_row.gt_x,gt_row.gt_y,gt_row.gt_z]
    pose_inicial = [odom_row.omom_px, odom_row.omom_py]

    Δx = gnss_inicial[0] - pose_inicial[0]
    Δy = gnss_inicial[1] - pose_inicial[1]

    ARRAY_Δx.append(Δx)
    ARRAY_Δy.append(Δy)
    ARRAY_date.append(np.unique(df_final.index.date))

    gt = gt[closest_gt_idx:]

    GNSScomprobacionX,GNSScomprobacionY=From_ODOM_2_GNSS(odom.omom_px.values, odom.omom_py.values)
    ODOMcomprobacionX,ODOMcomprobacionY=From_GNSS_2_ODOM(gt.gt_x.values, gt.gt_y.values)



    # ---------- DISCREPANCY ----------
    df_final["Discrepancy"] = np.sqrt(
        (trasladar_serie(df_final["omom_px"], Δx) - df_final["gt_x"])**2 +
        (trasladar_serie(df_final["omom_py"], Δy)- df_final["gt_y"])**2
    )
    # LABELS To Predict
    df_final["error_x"] = (trasladar_serie(df_final["gt_x"], -Δx)).values - df_final["omom_px"]
    df_final["error_y"] = (trasladar_serie(df_final["gt_y"], -Δy)).values - df_final["omom_py"]
    # ---------- SLIP INDEX ----------
    df_final["SV"] =  100*((df_final["cmd_vel_lx"].shift(1) - df_final["omom_tlx"]) /df_final["cmd_vel_lx"].shift(1))
    df_final["SV"] = df_final["SV"].bfill()
    # eliminar Picos
    df_final["SV"]=df_final["SV"].clip(-75,75)
    dt = df_final.index.to_series().diff().dt.total_seconds().mean()
    fs = 1 / dt
    print("dt =", dt, "segundos")
    print("fs =", fs, "Hz")
    dts = df_final.index.to_series().diff().dt.total_seconds()
    print("std dt:", dts.std())

    #plt.plot(df_final["SV"].values,label='Antes')
    df_final["SV"] = savgol_filter(df_final["SV"].values, 13, 3)
    #plt.plot(df_final["SV"].values,label='Despues')
    #plt.legend()
    #plt.show(block=True)

    df_final["error_x"] =(trasladar_serie(df_final["gt_x"],-Δx)).values-  df_final["omom_px"]
    df_final["error_y"] =(trasladar_serie(df_final["gt_y"],-Δy)).values-  df_final["omom_py"]
    # ---------- FEATURES ----------
    df_final["lin_vel"] = np.sqrt(
        df_final["omom_tlx"]**2 +0
        #df_final["omom_tly"]**2 +
        #df_final["omom_tlz"]**2
    )
    df_final["odom_acc_norm"] = np.sqrt(
        #df_final["omom_tax"]**2 +
        #df_final["omom_tay"]**2 +
        df_final["omom_taz"]**2
    )
    df_final["imu1_acc_norm"] = np.sqrt(
        df_final["vectorNAv_ax"]**2 +
        df_final["vectorNAv_ay"]**2 +
        df_final["vectorNAv_az"]**2
    )
    df_final["imu1_gyro_norm"] = np.sqrt(
        df_final["vectorNAv_wx"]**2 +
        df_final["vectorNAv_wy"]**2 +
        df_final["vectorNAv_wz"]**2
    )
    df_final["imu1_gyro_norm"] = np.sqrt(
        df_final["vectorNAv_gBODY_X"] ** 2 +
        df_final["vectorNAv_gBODY_Y"] ** 2 +
        df_final["vectorNAv_gBODY_Z"] ** 2
    )
    df_final["imu2_acc_norm"] = np.sqrt(
        df_final["xsens_ax"] ** 2 +
        df_final["xsens_ay"] ** 2 +
        df_final["xsens_az"] ** 2
    )
    df_final["imu2_gyro_norm"] = np.sqrt(
        df_final["xsens_gBODY_X"] ** 2 +
        df_final["xsens_gBODY_Y"] ** 2 +
        df_final["xsens_gBODY_Z"] ** 2
    )
    df_final["imu2_gyro_norm"] = np.sqrt(
        df_final["xsens_wx"] ** 2 +
        df_final["xsens_wy"] ** 2 +
        df_final["xsens_wz"] ** 2
    )
    df_final['LIN_VEL_x']=df_final['omom_LIN_VEL_x']
    df_final['LIN_VEL_y'] = df_final['omom_LIN_VEL_y']
    df_final['LIN_VEL_z'] = df_final['omom_LIN_VEL_z']

    df_final['ANG_VEL_x'] = df_final['omom_ANG_VEL_x']
    df_final['ANG_VEL_x'] = df_final['omom_ANG_VEL_x']
    df_final['ANG_VEL_x'] = df_final['omom_ANG_VEL_x']

    df_final['GT_LIN_VEL_x'] = df_final['LIN_VEL_x']
    df_final['GT_LIN_VEL_y'] = df_final['LIN_VEL_y']
    df_final['GT_LIN_VEL_z'] = df_final['LIN_VEL_z']

    df_final['GT_ANG_VEL_x'] = df_final['ANG_VEL_x']
    df_final['GT_ANG_VEL_x'] = df_final['ANG_VEL_x']
    df_final['GT_ANG_VEL_x'] = df_final['ANG_VEL_x']

    df_final["season"] = season
    return df_final


def FEATURE_IMP(input,flag_trin_test):
    print("\nTraining independent models independientes...")
    df_winter = input["winter"]
    df_spring = input["spring"]
    df_summer = input["summer"]
    df_autumn = input["autumn"]

    model_summer = train_model(df_summer)
    model_winter = train_model(df_winter)
    model_spring = train_model(df_spring)
    model_autumn = train_model(df_autumn)

    fi = pd.DataFrame(
        {
            "summer": model_summer.feature_importances_,
            "winter": model_winter.feature_importances_,
            "spring": model_spring.feature_importances_,
            "autumn": model_autumn.feature_importances_,
        },
        index=FEATURES_SV
    )
    print(FEATURES_SV)
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
    plt.tick_params(axis='both', labelsize=14)
    plt.show(block=True)


def tratamiento_XGB_season(input,test):
    print("\n--- Cross-season generalization ---")
    df_winter = input["winter"]
    df_spring = input["spring"]
    df_summer = input["summer"]
    df_autumn = input["autumn"]

    X_summer = df_summer[FEATURES_SV].values
    y_summer = df_summer[LABEL_SV].values
    X_winter = df_winter[FEATURES_SV].values
    y_winter = df_winter[LABEL_SV].values
    X_spring = df_spring[FEATURES_SV].values
    y_spring = df_spring[LABEL_SV].values
    X_autumn = df_autumn[FEATURES_SV].values
    y_autumn = df_autumn[LABEL_SV].values

    df_TEST_winter = test["winter"]
    df_TEST_spring = test["spring"]
    df_TEST_summer = test["summer"]
    df_TEST_autumn = test["autumn"]

    X_TEST_summer = df_TEST_summer[FEATURES_SV].values
    y_test_summer_index=df_TEST_summer.index
    y_TEST_summer = df_TEST_summer[LABEL_SV].values
    X_TEST_winter = df_TEST_winter[FEATURES_SV].values
    y_test_winter_index = df_TEST_winter.index
    y_TEST_winter = df_TEST_winter[LABEL_SV].values
    X_TEST_spring = df_TEST_spring[FEATURES_SV].values
    y_test_spring_index = df_TEST_spring.index
    y_TEST_spring = df_TEST_spring[LABEL_SV].values
    X_TEST_autumn = df_TEST_autumn[FEATURES_SV].values
    y_test_autumn_index = df_TEST_autumn.index
    y_TEST_autumn = df_TEST_autumn[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_summer, X_winter, X_autumn, X_spring]
    LABELS_ENTRENAMIENTO = [y_summer, y_winter, y_autumn, y_spring]
    INDICES = [y_test_summer_index, y_test_winter_index,y_test_autumn_index, y_test_spring_index,]
    DATOS_TEST = [X_TEST_summer, X_TEST_winter, X_TEST_autumn, X_TEST_spring]
    LABELS_TEST = [y_TEST_summer, y_TEST_winter, y_TEST_autumn, y_TEST_spring]

    results = []
    prediction = []
    estaciones = ["summer", "winter", "autumn", "spring"]
    for est1, X_train, y_train, X_test, y_test in zip(
            estaciones,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Entrenando estación: {est1}")

        modelxgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        modelxgb.fit(X_train, y_train)

        contador=0
        for est2, X_test, y_test in zip(
                estaciones,
                DATOS_TEST,
                LABELS_TEST
        ):
            print(f"Predicion estación: {est2}")
            print("Train size:", X_train.shape)
            print("Test size:", X_test.shape)
            print(INDICES[contador].shape)

            prediction_xgb=modelxgb.predict(X_test)
            prediction.append(prediction_xgb)

            mae_pred = mean_absolute_error(y_test, prediction_xgb)

            print(f"Time series: Real {est1} vs Prediction {est2}")
            print(f"MAE : {mae_pred:.4f}")

            plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter(np.arange(0, len(y_test)), prediction_xgb, label="Prediction XGB", color='green',linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                'mae':mae_pred,
                'prediccion':prediction_xgb
            })
            contador+=1

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction


def tratamiento_SGDseason(input,test):
    print("\n--- Cross-season generalization ---")
    df_winter = input["winter"]
    df_spring = input["spring"]
    df_summer = input["summer"]
    df_autumn = input["autumn"]

    X_summer = df_summer[FEATURES_SV].values
    y_summer = df_summer[LABEL_SV].values
    X_winter = df_winter[FEATURES_SV].values
    y_winter = df_winter[LABEL_SV].values
    X_spring = df_spring[FEATURES_SV].values
    y_spring = df_spring[LABEL_SV].values
    X_autumn = df_autumn[FEATURES_SV].values
    y_autumn = df_autumn[LABEL_SV].values

    df_TEST_winter = test["winter"]
    df_TEST_spring = test["spring"]
    df_TEST_summer = test["summer"]
    df_TEST_autumn = test["autumn"]

    X_TEST_summer = df_TEST_summer[FEATURES_SV].values
    y_test_summer_index=df_TEST_summer.index
    y_TEST_summer = df_TEST_summer[LABEL_SV].values
    X_TEST_winter = df_TEST_winter[FEATURES_SV].values
    y_test_winter_index = df_TEST_winter.index
    y_TEST_winter = df_TEST_winter[LABEL_SV].values
    X_TEST_spring = df_TEST_spring[FEATURES_SV].values
    y_test_spring_index = df_TEST_spring.index
    y_TEST_spring = df_TEST_spring[LABEL_SV].values
    X_TEST_autumn = df_TEST_autumn[FEATURES_SV].values
    y_test_autumn_index = df_TEST_autumn.index
    y_TEST_autumn = df_TEST_autumn[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_summer, X_winter, X_autumn, X_spring]
    LABELS_ENTRENAMIENTO = [y_summer, y_winter, y_autumn, y_spring]
    INDICES = [y_test_summer_index, y_test_winter_index,y_test_autumn_index, y_test_spring_index,]
    DATOS_TEST = [X_TEST_summer, X_TEST_winter, X_TEST_autumn, X_TEST_spring]
    LABELS_TEST = [y_TEST_summer, y_TEST_winter, y_TEST_autumn, y_TEST_spring]

    results = []
    prediction = []
    estaciones = ["summer", "winter", "autumn", "spring"]
    for est1, X_train, y_train, X_test, y_test in zip(
            estaciones,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Entrenando estación: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=15)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag
        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        contador=0
        for est2, X_test, y_test in zip(
                estaciones,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=15)  # create_lags
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Predicion estación: {est2}")
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
            resultados_sgd_offline, prediccion_sgd_offline = evaluate_modelPH("SGDRegressor without PartialFit", sgd1,
                                                                              X_train, y_train, X_test, y_test, 0,
                                                                              online=False)
            prediccion_sgd_offlinenversa = y_scaler.inverse_transform(
                np.array(prediccion_sgd_offline).reshape(-1, 1)).flatten()
            prediction.append(prediccion_sgd_offlinenversa)

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


            plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(15, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter( np.arange(15, len(y_test)),prediccion_sgd_offlinenversa, label="Prediction SGD", color='turquoise', linewidth=1, s=1)
            plt.scatter(np.arange(15, len(y_test)), prediccion_sgd_onlinenversa, label="Prediction SGD Partial", color='b', linewidth=1, s=1)
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                "SGD": resultados_sgd_offline,
                "SGD PARtial": resultados_sgd_online
            })
            contador+=1

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction


def tratamiento_TKANseason(input,test):
    print("\n--- Cross-season generalization ---")
    df_winter = input["winter"]
    df_spring = input["spring"]
    df_summer = input["summer"]
    df_autumn = input["autumn"]

    X_summer = df_summer[FEATURES_SV].values
    y_summer = df_summer[LABEL_SV].values
    X_winter = df_winter[FEATURES_SV].values
    y_winter = df_winter[LABEL_SV].values
    X_spring = df_spring[FEATURES_SV].values
    y_spring = df_spring[LABEL_SV].values
    X_autumn = df_autumn[FEATURES_SV].values
    y_autumn = df_autumn[LABEL_SV].values

    df_TEST_winter = test["winter"]
    df_TEST_spring = test["spring"]
    df_TEST_summer = test["summer"]
    df_TEST_autumn = test["autumn"]

    X_TEST_summer = df_TEST_summer[FEATURES_SV].values
    y_test_summer_index=df_TEST_summer.index
    y_TEST_summer = df_TEST_summer[LABEL_SV].values
    X_TEST_winter = df_TEST_winter[FEATURES_SV].values
    y_test_winter_index = df_TEST_winter.index
    y_TEST_winter = df_TEST_winter[LABEL_SV].values
    X_TEST_spring = df_TEST_spring[FEATURES_SV].values
    y_test_spring_index = df_TEST_spring.index
    y_TEST_spring = df_TEST_spring[LABEL_SV].values
    X_TEST_autumn = df_TEST_autumn[FEATURES_SV].values
    y_test_autumn_index = df_TEST_autumn.index
    y_TEST_autumn = df_TEST_autumn[LABEL_SV].values

    DATOS_ENTRENAMIENTO = [X_summer, X_winter, X_autumn, X_spring]
    LABELS_ENTRENAMIENTO = [y_summer, y_winter, y_autumn, y_spring]
    INDICES = [y_test_summer_index, y_test_winter_index,y_test_autumn_index, y_test_spring_index,]
    DATOS_TEST = [X_TEST_summer, X_TEST_winter, X_TEST_autumn, X_TEST_spring]
    LABELS_TEST = [y_TEST_summer, y_TEST_winter, y_TEST_autumn, y_TEST_spring]

    results = []
    prediction = []
    estaciones = ["summer", "winter", "autumn", "spring"]
    for est1, X_train, y_train, X_test, y_test in zip(
            estaciones,
            DATOS_ENTRENAMIENTO,
            LABELS_ENTRENAMIENTO,
            DATOS_TEST,
            LABELS_TEST
    ):
        print(f"Entrenando estación: {est1}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = x_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        XTRAIN_lag, yTRAIN_lag = create_sequences_flat(X_train, y_train, lags=15)  # create_lags
        X_train, y_train = XTRAIN_lag, yTRAIN_lag

        contador=0
        for est2, X_test, y_test in zip(
                estaciones,
                DATOS_TEST,
                LABELS_TEST
        ):
            XTEST_lag, yTEST_lag = create_sequences_flat(X_test, y_test, lags=15)  # create_lags
            X_test, y_test = XTEST_lag, yTEST_lag
            X_test = x_scaler.transform(X_test)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            print(f"Predicion estación: {est2}")
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

            plt.figure(figsize=(12, 5))
            plt.scatter(np.arange(0, len(y_test)), y_test, label="Real", color='r', linewidth=1, s=1)
            plt.scatter(np.arange(0, len(y_test)), prediccionT_KAN_inversa, label="Prediction TKAN", color='lightblue', linewidth=1, s=1)#
            plt.legend()
            plt.title(f"Time series: Real {est1} vs Prediction {est2}")
            plt.xlabel("Timestamp")
            plt.ylabel('SV')
            plt.show(block=True)

            results.append({
                "train_dataset": est1,
                "test_dataset": est2,
                "T-KAN": resultadosT_KAN,
            })
            contador+=1

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction


def tratamiento_XGB_WHOLE(X,y,X_t,y_t):
    print("\n--- WHOLE DATASET ---")

    X_train = X[FEATURES_SV].values
    y_train = y.values
    X_test = X_t[FEATURES_SV].values
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


def tratamiento_SGD_WHOLE(X,y,X_t,y_t):
    print("\n--- WHOLE DATASET ---")

    X_train = X[FEATURES_SV].values
    y_train = y.values
    X_test = X_t[FEATURES_SV].values
    y_test = y_t.values
    indice = y_t.index

    results = []
    prediction = []

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    XTRAIN_lag, yTRAIN_lag = create_lags_fast(X_train, y_train, lags=15)  # create_lags
    X_train, y_train = XTRAIN_lag, yTRAIN_lag
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    XTEST_lag, yTEST_lag = create_lags_fast(X_test, y_test, lags=15)  # create_lags
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
        mask = (pd.to_datetime(indice[15:]).normalize() == fecha)
        maskdriftsgd=(pd.to_datetime(indice[15:][resultados_sgd_offline['drifts']]).normalize() == fecha)
        maskdriftsgdpartial = (pd.to_datetime(indice[15:][resultados_sgd_online['drifts']]).normalize() == fecha)

        horas = indice[15:][mask]
        reales = y_test[mask]
        pred1 = prediccion_sgd_offlinenversa[mask]
        pred2 = prediccion_sgd_onlinenversa[mask]
        drift1 = indice[15:][resultados_sgd_offline['drifts']][maskdriftsgd]
        drift2 = indice[15:][resultados_sgd_online['drifts']][maskdriftsgdpartial]

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
        plt.ylabel('SV')
        plt.show(block=True)

    results.append({
        "SGD": resultados_sgd_offline,
        "SGD PARtial": resultados_sgd_online
    })

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)

    return prediction,resultados_sgd_offline['drifts'],resultados_sgd_online['drifts']


def tratamiento_TKAN_WHOLE(X,y,X_t,y_t):
    print("\n--- WHOLE DATASET ---")
    X_train = X[FEATURES_SV].values
    y_train = y.values
    X_test = X_t[FEATURES_SV].values
    y_test = y_t.values
    indice = y_t.index

    results = []
    prediction = []

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    XTRAIN_lag, yTRAIN_lag = create_sequences_flat(X_train, y_train, lags=15)  # create_lags
    X_train, y_train = XTRAIN_lag, yTRAIN_lag

    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    XTEST_lag, yTEST_lag = create_sequences_flat(X_test, y_test, lags=15)  # create_lags
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
        mask = (pd.to_datetime(indice[15:]).normalize() == fecha)
        maskdriftsgd = (pd.to_datetime(indice[15:][resultadosT_KAN['drifts']]).normalize() == fecha)

        horas = indice[15:][mask]
        reales = y_test[mask]
        pred = prediccionT_KAN_inversa[mask]
        drift1 = indice[15:][resultadosT_KAN['drifts']][maskdriftsgd]

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


def train_model(df):
   X = df[FEATURES_SV]
   y = df["SV"]
   model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42)
   model.fit(X, y)
   return model


def TRAIN_LINEARvelocity(input):
    models=[]
    return models


def TRAIN_ANGULARvelocity(input):
    models=[]
    return models
