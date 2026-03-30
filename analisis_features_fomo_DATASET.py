import pandas as pd
pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from PH import PageHinkley
import os
from pathlib import Path
from datetime import datetime
from analisis_features_utils import (tratamiento_XGB_WHOLE,tratamiento_SGD_WHOLE,tratamiento_TKAN_WHOLE,
                                     load_meta,load_trajectory_files,FEATURE_IMP)

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

features_terrain=['Soil Condition']#'Slope'== INFO GRAVITY

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
lABEL_L_W=['GT_LIN_VEL_x',
    'GT_LIN_VEL_y',
    'GT_LIN_VEL_z',
    'GT_ANG_VEL_x',
    'GT_ANG_VEL_y',
    'GT_ANG_VEL_z']

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
def recorrer_fechas(ruta_base):
    train = {"winter": [], "spring": [], "summer": [], "autumn": []}
    test = {"winter": [], "spring": [], "summer": [], "autumn": []}
    train_final = {}
    test_final = {}

    for fecha_dir in os.listdir(ruta_base):
        ruta_fecha = Path(ruta_base) / fecha_dir
        if not ruta_fecha.is_dir():
            continue
        season = get_season(fecha_dir)
        print(f"\n {fecha_dir} → {season}")
        es_train = fecha_dir in fechas_train
        es_test = fecha_dir in fechas_test
        for subdir in ruta_fecha.iterdir():
            if subdir.is_dir() and subdir.name.startswith("red"):
                df_meteo, df_imu = analizar_directorio(subdir,season)
                df_imu["fecha"] = fecha_dir
                df_meteo = df_meteo.drop(columns=["season"], errors="ignore")
                #MERGE
                if  df_meteo is None or df_meteo.empty:
                    print("⚠️ Meteo vacío, se devuelve IMU solo")
                    df_merged = df_imu.copy()
                else:
                    df_merged = pd.merge_asof(
                        df_imu.sort_index(),
                        df_meteo.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="nearest",
                        tolerance=pd.Timedelta("2min")
                    )
                if es_test:
                    test[season].append(df_merged)
                else:
                    train[season].append(df_merged)

    for season in ["winter", "spring", "summer", "autumn"]:
        # TRAIN
        if train[season]:
            train_final[season] = pd.concat(train[season])
        # TEST
        if test[season]:
            test_final[season] = pd.concat(test[season])
    df_train_all = pd.concat(train_final.values(), axis=0)
    df_test_all = pd.concat(test_final.values(), axis=0)
    return train_final,test_final,df_train_all, df_test_all
def analizar_directorio(ruta_red,season):
    METEO_PATH = ruta_red / "metadata"

    print("METADATA...")
    df_meta = load_meta(METEO_PATH,season)

    print("Loading data...")
    df_traj = load_trajectory_files(ruta_red,season)

    print("METADATA: ", df_meta.shape)
    print("TRAJECTORY DATA: ", df_traj.shape)

    return df_meta, df_traj

if __name__ == '__main__':
    # ============================================================
    # LOAD DATA
    # ============================================================
    if( (os.path.exists(NEW_FOMO_PATH+"springtest.csv")) & (os.path.exists(NEW_FOMO_PATH+"DATASET_TEST.csv")) & (os.path.exists(NEW_FOMO_PATH+"DATASET_TRAIN.csv"))):
        print("El archivo existe")
        datasettrain = pd.read_csv(NEW_FOMO_PATH + "DATASET_TRAIN.csv")
        datasettest = pd.read_csv(NEW_FOMO_PATH + "DATASET_TEST.csv")
        # Use as index
        datasettrain = datasettrain.set_index("t")
        datasettest = datasettest.set_index("t")

        estacion = ["summer", "winter", "autumn", "spring"]

        train_season = {}
        test_season = {}
        # Reconstruir train
        for est in estacion:
            file_path = os.path.join(NEW_FOMO_PATH, f"{est}train.csv")
            train_season[est] = pd.read_csv(file_path, index_col=0)

        # Reconstruir test
        for est in estacion:
            file_path = os.path.join(NEW_FOMO_PATH, f"{est}test.csv")
            test_season[est] = pd.read_csv(file_path, index_col=0)
    else:
        print("El archivo NO existe")
        train_season,test_season,datasettrain, datasettest = recorrer_fechas(BASE_PATH)
        datasettrain.to_csv(NEW_FOMO_PATH + "DATASET_TRAIN.csv", index=True)
        datasettest.to_csv(NEW_FOMO_PATH + "DATASET_TEST.csv", index=True)

        estacion = ["summer", "winter", "autumn", "spring"]

        for estacion, df_est in train_season.items():
            df_est.to_csv(NEW_FOMO_PATH + f"{estacion}train.csv", index=True)
        for estacion, df_est in test_season.items():
            df_est.to_csv(NEW_FOMO_PATH + f"{estacion}test.csv", index=True)

    print("\n✅ Dataset1:", len(datasettrain))
    print("✅ Dataset2:", len(datasettest))

    print('NAN ',datasettrain.isna().sum())
    print('NAN  ',datasettest.isna().sum())

    datasettrain = datasettrain.fillna(-1)
    datasettest = datasettest.fillna(-1)

    #PREPROCESS Slip INDEX
    pred_SV=[]

    X_train = datasettrain[FEATURES_SV]
    datasettrain.index = pd.to_datetime(datasettrain.index)
    y_train = datasettrain["SV"]
    X_train_gt = datasettrain[["gt_x", "gt_y"]]

    X_test = datasettest[FEATURES_SV]
    datasettest.index = pd.to_datetime(datasettest.index)
    y_test = datasettest["SV"]
    X_test_gt = datasettest[["gt_x", "gt_y"]]

    # SIGNAL ANALISIS
    # FEATURE_IMP(train_season,0)#0 for train
    # FEATURE_IMP(test_season,1)#1 for test

    '''summer = datasettrain[datasettrain['season'] == 'summer']["SV"]
    winter = datasettrain[datasettrain['season'] == 'winter']["SV"]
    autumn = datasettrain[datasettrain['season'] == 'autumn']["SV"]
    spring = datasettrain[datasettrain['season'] == 'spring']["SV"]

    summer_mean = np.mean(summer)
    print('MEDIA SUMMER ', summer_mean)
    winter_mean = np.mean(winter)
    print('MEDIA WINTER ', winter_mean)
    autumn_mean = np.mean(autumn)
    print('MEDIA AUTUMN ', autumn_mean)
    spring_mean = np.mean(spring)
    print('MEDIA  SPRING ', spring_mean)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(summer, label="Slip Index  for the TRAIN summer", color='gold', linewidth=2)
    ax.set_title("TRAIN SUMMER")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = np.max(summer.values)
    y_min = np.min(summer.values)
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = summer.index[int(len(summer) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(winter, label="Slip Index  for the TRAIN winter", color='blue', linewidth=2)
    ax.set_title("TRAIN WINTER")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = winter.values.max()
    y_min = winter.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = winter.index[
        int(len(winter) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(spring, label="Slip Index  for the TRAIN spring", color='green', linewidth=2)
    ax.set_title("TRAIN SPRING")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = spring.values.max()
    y_min = spring.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = spring.index[
        int(len(spring) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(autumn, label="Slip Index  for the TRAIN autumn", color='orange', linewidth=2)
    ax.set_title("TRAIN AUTUMN")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = autumn.values.max()
    y_min = autumn.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = autumn.index[
        int(len(autumn) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    summer = datasettest[datasettest['season'] == 'summer']["SV"]
    winter = datasettest[datasettest['season'] == 'winter']["SV"]
    autumn = datasettest[datasettest['season'] == 'autumn']["SV"]
    spring = datasettest[datasettest['season'] == 'spring']["SV"]

    summer_mean = np.mean(summer)
    print('MEDIA SUMMER ', summer_mean)
    winter_mean = np.mean(winter)
    print('MEDIA WINTER ', winter_mean)
    autumn_mean = np.mean(autumn)
    print('MEDIA AUTUMN ', autumn_mean)
    spring_mean = np.mean(spring)
    print('MEDIA  SPRING ', spring_mean)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(summer, label="Slip Index  for the TEST summer", color='gold', linewidth=2)
    ax.set_title("TEST SUMMER")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = np.max(summer.values)
    y_min = np.min(summer.values)
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = summer.index[int(len(summer) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(winter, label="Slip Index  for the TEST winter", color='blue', linewidth=2)
    ax.set_title("TEST WINTER")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = winter.values.max()
    y_min = winter.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = winter.index[
        int(len(winter) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(spring, label="Slip Index  for the TEST spring", color='green', linewidth=2)
    ax.set_title("TEST SPRING")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = spring.values.max()
    y_min = spring.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = spring.index[
        int(len(spring) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(autumn, label="Slip Index  for the TEST autumn", color='orange', linewidth=2)
    ax.set_title("TEST AUTUMN")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = autumn.values.max()
    y_min = autumn.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = autumn.index[
        int(len(autumn) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)'''

    #MODELS TO DETECT SV CD
    #XGBpred=tratamiento_XGB_WHOLE(X_train,y_train,X_test,y_test)

    predictionSGD, SGDFITdrifts,SGDPartialdrifts=tratamiento_SGD_WHOLE(X_train, y_train, X_test, y_test)
    print('Drift detectados por SGD: ',datasettest.index[SGDFITdrifts])
    print('Drift detectados por Partial SGD: ',datasettest.index[SGDPartialdrifts])

    TKANpred,TKANdrfts=tratamiento_TKAN_WHOLE(X_train, y_train, X_test, y_test)
    print('Drift detectados por TKAN: ', datasettest.index[TKANdrfts])

    #Now we should train VAR modes according the different classifications of SV

    '''
    pred_errores=[]
    
    X_train = datasettrain[FEATURES_L_W]
    y_train1 = datasettrain["error_lin"]
    y_train1 = datasettrain["error_and"]
    X_train_gt = datasettrain[["gt_x", "gt_y"]]

    X_test = datasettest[FEATURES_L_W]
    y_test1 = datasettest["error_lin"]
    y_test2 = datasettest["error_ang"]
    X_test_gt = datasettest[["gt_x", "gt_y"]]
    
    FEATURE_IMP(train_season)
    FEATURE_IMP(test_season)

    predicciones = tratamiento_season(train_season, test_season, flag)
    pred_errores.append(predicciones)

    VISUALIZACION_MEJORA(train_season,test_season,pred_errores, 'PREDICTION WITH SV')'''

    print('FIN')

