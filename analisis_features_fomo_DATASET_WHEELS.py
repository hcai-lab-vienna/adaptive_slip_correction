import pandas as pd

from BOKU.analisis_features_utils import tratamiento_SGDPARTIALseason

pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from pathlib import Path
from datetime import datetime
from analisis_features_utils import (tratamiento_XGB_WHOLE,tratamiento_SGD_WHOLE,tratamiento_TKAN_WHOLE,
                                     load_meta,load_trajectory_data2,FEATURE_IMP,tratamiento_SGDseason,
                                     tratamiento_XGB_season,VISUALIZACION_MEJORA)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "..\\..\\..\\OSR\\code\\fomo-dataset"
NEW_FOMO_PATH = "..\\..\\..\\OSR\\code\\New_fomo_DATASET\\"

fechas_train = {"2025-04-15","2025-09-24","2024-11-21","2025-08-20", "2025-10-14","2024-11-28"}#"2025-06-26",
fechas_test = { "2025-05-28","2025-11-03"}
fecha_SNOW_road_test = {"2024-11-28"}
CONDITIONS_DATE=[ ("2025-04-15",3),("2025-06-26",2),("2025-09-24",2),("2025-11-03",3),("2024-11-21",2),("2025-05-28",2),("2025-08-20",2),("2025-10-14",2),("2024-11-28",1)]
CONDITION_CLASSIFICATION_EXPLAINATION=[(1,'snow on the road, not snowing'),
                          (2,'clear road, not raining'),
                          (3,'clear road, raining')]
CONDITION_CLASSIFICATION=[(1,'snow_road'),
                          (2,'clear_road'),
                          (3,'clear_raining')]

FEATURES_METEO = [
    'Rain_accum',
    'RH',
    'SnowDepth_Avg',
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
    'ang_vel_cmd_z',
    'SV'
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
    'ang_vel_cmd_z',
    'SV'
]
lABEL_L_W='TARGET'# GT LINEAR VELOCITY

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
    # Dictionary: DATE -> condition
    date_to_condition = dict(CONDITIONS_DATE)

    train_condition = {}
    test_condition = {}

    train_final = {}
    test_final = {}

    # inicializar listas
    for cond_id in set(date_to_condition.values()):
        train_condition[cond_id] = []
        test_condition[cond_id] = []

    for fecha_dir in os.listdir(ruta_base):
        ruta_fecha = Path(ruta_base) / fecha_dir
        if not ruta_fecha.is_dir():
            continue
        season = get_season(fecha_dir)
        print(f"\n {fecha_dir} → {season}")

        es_train = fecha_dir in fechas_train
        es_test = fecha_dir in fechas_test

        # obtener condición desde el mapping
        cond_id = date_to_condition.get(fecha_dir)

        for subdir in ruta_fecha.iterdir():
            if subdir.is_dir() and subdir.name.startswith("red"):
                df_meteo, df_imu = analizar_directorio(subdir,season,fecha_dir)

                #MERGE
                if  df_meteo is None or df_meteo.empty:
                    print("⚠️ Meteo vacío, se devuelve IMU solo")
                    df_merged = df_imu.copy()
                else:
                    df_imurounded = df_imu.index.round('min')
                    df_meteorounded = df_meteo.index.round('min')

                    hay_interseccion = (
                            df_imurounded.min() <= df_meteorounded.max() and
                            df_meteorounded.min() <= df_imurounded.max()
                    )

                    print("Hay intersección:", hay_interseccion)
                    print("IMU:", df_imu.index.min(), "->", df_imu.index.max())
                    print("METEO:", df_meteo.index.min(), "->", df_meteo.index.max())

                    df_merged = pd.merge_asof(
                        df_imu.sort_index(),
                        df_meteo.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="nearest",
                        tolerance=pd.Timedelta("2min")#1min
                    )
                    hay_nan = df_merged.isna().any().any()
                    print("⚠️ Hay NaN:", hay_nan)

                    if hay_nan:
                        print("NaN por columna:")
                        print(df_merged.isna().sum())

                # ---------- SLIP INDEX ----------
                no_ceros = df_merged[(df_merged["lin_vel_odom_x"] != 0)&(df_merged["TARGET"] != 0)]
                no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0)]
                start = no_ceros.index[0]
                end = no_ceros.index[-1]
                df_merged = df_merged.loc[start:end]
                df_merged["SV"] = 100 * (
                            (df_merged["TARGET"] - df_merged["lin_vel_odom_x"]) / df_merged[
                        "TARGET"])
                df_merged["SV"] = 100 * (
                        (df_merged["TARGET"] - df_merged["lin_vel_cmd_x"]) / df_merged[
                    "TARGET"])
                #df_merged["SV"] = df_merged["SV"].bfill()

                # diferencia absoluta entre puntos consecutivos
                diff = df_merged["SV"].diff().abs()
                # índices donde hay salto
                threshold = 40
                # cambio de signo entre t-1 y t
                sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
                # magnitud del salto
                big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
                # spikes finales
                spikes = sign_change & big_jump
                indices_spikes = df_merged["SV"][spikes].index
                # eliminar Picos
                df_merged["SV"] = df_merged["SV"].clip(-75, 75)
                #plt.plot(df_merged["SV"])
                #plt.show(block=True)
                df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
                #plt.plot(df_merged["SV"])
                #plt.show(block=True)
                mask_bad = ~np.isfinite(df_merged['SV'])
                df_merged = df_merged[~mask_bad].copy()

                # Features to build 'NORMA_lin_acc_imu', 'NORMA_ang_vel_imu'
                df_merged["NORMA_lin_acc_imu"] =np.sqrt(df_merged['lin_acc_imu_x']**2 + df_merged['lin_acc_imu_y']**2 + df_merged['lin_acc_imu_z']**2)
                df_merged["NORMA_ang_vel_imu"] =np.sqrt(df_merged['ang_vel_imu_x']**2 + df_merged['ang_vel_imu_y']**2 + df_merged['ang_vel_imu_z']**2)
                df_merged["lin_vel_twist"] =df_merged["lin_vel_odom_x"]
                df_merged["ang_vel_twist"] =df_merged['ang_vel_odom_z']

                if es_train:
                    if hay_nan == False:
                        train_condition[cond_id].append(df_merged)
                        print('TRAIN No nan',fecha_dir)
                if es_test:
                    if hay_nan == False:
                        test_condition[cond_id].append(df_merged)
                        print('TEST No nan', fecha_dir)

    #extratrajectory for snow_road for test
    ruta=Path(ruta_base) / '2024-11-28'
    for subdir in ruta.iterdir():
        if subdir.is_dir() and subdir.name.startswith("yellow"):
            df_meteo, df_imu = analizar_directorio(subdir, 'autumn', '2024-11-28',color="yellow")
            # MERGE
            if df_meteo is None or df_meteo.empty:
                print("⚠️ Meteo vacío, se devuelve IMU solo")
                df_merged = df_imu.copy()
            else:
                df_imurounded = df_imu.index.round('min')
                df_meteorounded = df_meteo.index.round('min')

                hay_interseccion = (
                        df_imurounded.min() <= df_meteorounded.max() and
                        df_meteorounded.min() <= df_imurounded.max()
                )
                print("Hay intersección:", hay_interseccion)
                print("IMU:", df_imu.index.min(), "->", df_imu.index.max())
                print("METEO:", df_meteo.index.min(), "->", df_meteo.index.max())

                df_merged = pd.merge_asof(
                    df_imu.sort_index(),
                    df_meteo.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="nearest",
                    tolerance=pd.Timedelta("2min")  # 1min
                )
                hay_nan = df_merged.isna().any().any()
                print("⚠️ Hay NaN:", hay_nan)

                if hay_nan:
                    print("NaN por columna:")
                    print(df_merged.isna().sum())

            # ---------- SLIP INDEX ----------
            no_ceros = df_merged[(df_merged["lin_vel_odom_x"] != 0) & (df_merged["TARGET"] != 0)]
            no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0)]
            start = no_ceros.index[0]
            end = no_ceros.index[-1]
            df_merged = df_merged.loc[start:end]
            df_merged["SV"] = 100 * (
                    (df_merged["TARGET"] - df_merged["lin_vel_odom_x"]) / df_merged[
                "TARGET"])
            df_merged["SV"] = 100 * (
                    (df_merged["TARGET"] - df_merged["lin_vel_cmd_x"]) / df_merged[
                "TARGET"])
            # df_merged["SV"] = df_merged["SV"].bfill()
            # diferencia absoluta entre puntos consecutivos
            diff = df_merged["SV"].diff().abs()
            # índices donde hay salto
            threshold = 40
            # cambio de signo entre t-1 y t
            sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
            # magnitud del salto
            big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
            # spikes finales
            spikes = sign_change & big_jump
            indices_spikes = df_merged["SV"][spikes].index
            # eliminar Picos
            df_merged["SV"] = df_merged["SV"].clip(-75, 75)
            # plt.plot(df_merged["SV"])
            # plt.show(block=True)
            df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
            # plt.plot(df_merged["SV"])
            # plt.show(block=True)
            mask_bad = ~np.isfinite(df_merged['SV'])
            df_merged = df_merged[~mask_bad].copy()

            # Features to build 'NORMA_lin_acc_imu', 'NORMA_ang_vel_imu'
            df_merged["NORMA_lin_acc_imu"] = np.sqrt(
                df_merged['lin_acc_imu_x'] ** 2 + df_merged['lin_acc_imu_y'] ** 2 + df_merged['lin_acc_imu_z'] ** 2)
            df_merged["NORMA_ang_vel_imu"] = np.sqrt(
                df_merged['ang_vel_imu_x'] ** 2 + df_merged['ang_vel_imu_y'] ** 2 + df_merged['ang_vel_imu_z'] ** 2)
            df_merged["lin_vel_twist"] = df_merged["lin_vel_odom_x"]
            df_merged["ang_vel_twist"] = df_merged['ang_vel_odom_z']

            if hay_nan==False:
                test_condition[1].append(df_merged)
                print('yellow','2024-11-28')

    for cond in date_to_condition.values():
        # TRAIN
        if train_condition[cond]:
            train_final[cond] = pd.concat(train_condition[cond])
        # TEST
        if test_condition[cond]:
            test_final[cond] = pd.concat(test_condition[cond])

    df_train_all = pd.concat(train_final.values(), axis=0)
    df_test_all = pd.concat(test_final.values(), axis=0)

    return train_final,test_final,df_train_all, df_test_all
def analizar_directorio(ruta_red,season,fecha,color="red"):
    print("Loading data...")
    df_traj = load_trajectory_data2(fecha,color)

    METEO_PATH = ruta_red / "metadata"

    print("METADATA...")
    df_meta = load_meta(METEO_PATH, season)

    print("METADATA: ", df_meta.shape)
    print("TRAJECTORY DATA: ", df_traj.shape)
    print("TRAJECTORY Index: ", len(df_traj.index))

    return df_meta, df_traj

if __name__ == '__main__':
    # ============================================================
    # LOAD DATA
    # ============================================================
    if( (os.path.exists(NEW_FOMO_PATH+"snow_roadtrain.csv")) & (os.path.exists(NEW_FOMO_PATH+"DATASET_TEST.csv")) & (os.path.exists(NEW_FOMO_PATH+"DATASET_TRAIN.csv"))):
        print("El archivo existe")
        datasettrain = pd.read_csv(NEW_FOMO_PATH + "DATASET_TRAIN.csv")
        datasettest = pd.read_csv(NEW_FOMO_PATH + "DATASET_TEST.csv")
        # Use as index
        datasettrain = datasettrain.set_index("Unnamed: 0")
        datasettest = datasettest.set_index("Unnamed: 0")

        train_condition = {}
        test_condition = {}
        for cond_id, cond_name in CONDITION_CLASSIFICATION:
            file_path = os.path.join(NEW_FOMO_PATH, f"{cond_name}train.csv")
            train_condition[cond_id] = pd.read_csv(file_path, index_col=0)

        # Construir test
        for cond_id, cond_name in CONDITION_CLASSIFICATION:
            file_path = os.path.join(NEW_FOMO_PATH, f"{cond_name}test.csv")
            test_condition[cond_id] = pd.read_csv(file_path, index_col=0)
    else:
        print("El archivo NO existe")
        train_condition,test_condition,datasettrain, datasettest = recorrer_fechas(BASE_PATH)
        datasettrain.to_csv(NEW_FOMO_PATH + "DATASET_TRAIN.csv", index=True)
        datasettest.to_csv(NEW_FOMO_PATH + "DATASET_TEST.csv", index=True)

        all_conds = sorted(set(train_condition.keys()) | set(test_condition.keys()))
        for cond in all_conds:
            # Train
            if cond in train_condition:
                df_tr = train_condition[cond]
                df_tr.to_csv(
                    NEW_FOMO_PATH + f"{CONDITION_CLASSIFICATION[cond - 1][1]}train.csv",
                    index=True
                )

            # Test
            if cond in test_condition:
                df_est = test_condition[cond]
                df_est.to_csv(
                    NEW_FOMO_PATH + f"{CONDITION_CLASSIFICATION[cond - 1][1]}test.csv",
                    index=True
                )

    print("\n✅ Dataset1:", len(datasettrain))
    print("✅ Dataset2:", len(datasettest))

    print('NAN ',datasettrain.isna().sum())
    print('NAN  ',datasettest.isna().sum())

    datasettrain = datasettrain.fillna(-1)
    datasettest = datasettest.fillna(-1)

    #PREPROCESS Slip INDEX
    pred_SV=[]
    pred_LIN_VEL=[]
    FEATURES1 = FEATURES_METEO + FEATURES_SV1
    FEATURES2 = FEATURES_METEO + FEATURES_SV2
    POSIBILIDAD_FEATURES=[FEATURES1,FEATURES2]
    POSIBILIDAD_FEATURES = [FEATURES1]
    length_fetauresSV=[len(FEATURES1),len(FEATURES2)]
    length_fetauresSV = [len(FEATURES1) ]

    #ANALISIS POR CONDICIONES
    for opt in POSIBILIDAD_FEATURES:
        X_train = datasettrain[opt]
        datasettrain.index = pd.to_datetime(datasettrain.index)
        y_train = datasettrain["SV"]

        X_test = datasettest[opt]
        datasettest.index = pd.to_datetime(datasettest.index)
        y_test = datasettest["SV"]

        '''tratamiento_XGB_season(train_condition,test_condition,opt,'SV')
        tratamiento_SGDseason(train_condition,test_condition,opt,'SV')
        tratamiento_SGDPARTIALseason(train_condition, test_condition, opt, 'SV')'''


        predictionSGD, SGDFITdrifts, SGDPartialdrifts = tratamiento_SGD_WHOLE(X_train, y_train, X_test, y_test, opt,"SV")

        print('Drift detectados por SGD: ', datasettest.index[SGDFITdrifts])
        print('Drift detectados por Partial SGD: ', datasettest.index[SGDPartialdrifts])
        pred_SV.append(predictionSGD[1])
        # TKANpred,TKANdrfts=tratamiento_TKAN_WHOLE(X_train, y_train, X_test, y_test,opt)
        # print('Drift detectados por TKAN: ', datasettest.index[TKANdrfts])

    snow_road = datasettrain[datasettrain['Soil_type'] == 1]["SV"]
    clear_road = datasettrain[datasettrain['Soil_type'] == 2]["SV"]
    clear_raining = datasettrain[datasettrain['Soil_type'] == 3]["SV"]

    snow_road.index = pd.to_datetime(snow_road.index)
    clear_road.index = pd.to_datetime(clear_road.index)
    clear_raining.index = pd.to_datetime(clear_raining.index)

    snow_road_mean = np.mean(snow_road)
    print('MEDIA snow_road_mean ', snow_road_mean)
    clear_road_mean = np.mean(clear_road)
    print('MEDIA clear_road_mean ', clear_road_mean)
    clear_rainig_mean = np.mean(clear_raining)
    print('MEDIA  clear_rainig_mean ', clear_rainig_mean)

    '''
    # Agrupar por fecha (día)
    for fecha, grupo in snow_road.groupby(snow_road.index.date):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)

        ax.plot(grupo, label=f"Slip Index {fecha}", color='gold', linewidth=2)

        ax.set_title(f"TRAIN snow_road - {fecha}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Slip index")

        y_max = np.max(grupo.values)
        y_min = np.min(grupo.values)

        ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
        ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)

        x_pos = grupo.index[int(len(grupo) * 0.85)]

        ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
        ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)

        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    
    # Agrupar por fecha (día)
    for fecha, grupo in clear_road.groupby(clear_road.index.date):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)

        ax.plot(grupo, label=f"Slip Index {fecha}", color='green', linewidth=2)

        ax.set_title(f"TRAIN clear_road - {fecha}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Slip index")

        y_max = np.max(grupo.values)
        y_min = np.min(grupo.values)

        ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
        ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)

        x_pos = grupo.index[int(len(grupo) * 0.85)]

        ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
        ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)

        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    # Agrupar por fecha (día)
    for fecha, grupo in clear_raining.groupby(clear_raining.index.date):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)

        ax.plot(grupo, label=f"Slip Index {fecha}", color='orange', linewidth=2)

        ax.set_title(f"TRAIN clear_raining - {fecha}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Slip index")

        y_max = np.max(grupo.values)
        y_min = np.min(grupo.values)

        ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
        ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)

        x_pos = grupo.index[int(len(grupo) * 0.85)]

        ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
        ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)

        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
        
    '''
    snow_road = datasettest[datasettest['Soil_type'] == 1]["SV"]
    clear_road = datasettest[datasettest['Soil_type'] == 2]["SV"]
    clear_raining = datasettest[datasettest['Soil_type'] == 3]["SV"]

    snow_road_mean = np.mean(snow_road)
    print('MEDIA snow_road_mean ', snow_road_mean)
    clear_road_mean = np.mean(clear_road)
    print('MEDIA clear_road_mean ', clear_road_mean)
    clear_rainig_mean = np.mean(clear_raining)
    print('MEDIA  clear_rainig_mean ', clear_rainig_mean)

    '''fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(snow_road, label="Slip Index  for the TEST snow_road", color='gold', linewidth=2)
    ax.set_title("TEST snow_road")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = np.max(snow_road.values)
    y_min = np.min(snow_road.values)
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = snow_road.index[int(len(snow_road) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(clear_road, label="Slip Index  for the TEST clear_road", color='green', linewidth=2)
    ax.set_title("TEST clear_road")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = clear_road.values.max()
    y_min = clear_road.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = clear_road.index[
        int(len(clear_road) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(clear_raining, label="Slip Index  for the TEST clear_raining", color='orange', linewidth=2)
    ax.set_title("TEST clear_raining")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Slip index")
    y_max = clear_raining.values.max()
    y_min = clear_raining.values.min()
    ax.axhline(y_max, color='red', linestyle='--', linewidth=1)
    ax.axhline(y_min, color='blue', linestyle='--', linewidth=1)
    x_pos = clear_raining.index[
        int(len(clear_raining) * 0.85)]
    ax.text(x_pos, y_max, f'{y_max:.2f}', color='red', fontsize=8)
    ax.text(x_pos, y_min, f'{y_min:.2f}', color='blue', fontsize=8)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show(block=True)'''


    # Now we should train VAR modes according the different classifications of SV
    # Then the prediction of the SV, and the classification of the model
    # Then the prediction of the correction of linear and angular velocity according thr classifcation os SV and model.

    FEATURES1 = FEATURES_METEO + FEATURES_L_W_1
    FEATURES2 = FEATURES_METEO + FEATURES_L_W_2
    POSIBILIDAD_FEATURES_LIN = [FEATURES1, FEATURES2]
    POSIBILIDAD_FEATURES_LIN = [FEATURES1]

    # MODELS TO USING SV
    cont=0
    for opt in POSIBILIDAD_FEATURES_LIN:
        pred_errores=[]

        X_train = datasettrain[opt]
        y_train1 = datasettrain["TARGET"]

        #SUSTITIUMOS LA PREDICCION
        datasettest['SV'][length_fetauresSV[cont]:]=pred_SV[cont]
        X_test = datasettest[opt]
        y_test1 = datasettest["TARGET"]

        tmp=0
        for key, df in test_condition.items():
            if key==1:
                df.loc[df.index[length_fetauresSV[cont]:], 'SV'] = pred_SV[cont][:df.loc[df.index[length_fetauresSV[cont]:], 'SV'].shape[0]]
                tmp=df.loc[df.index[length_fetauresSV[cont]:], 'SV'].shape[0]
            elif key==2:
                df.loc[df.index[:], 'SV'] = pred_SV[cont][tmp:df.shape[0]+tmp]
                tmp+= df.shape[0]
            else:
                df.loc[df.index[:], 'SV'] = pred_SV[cont][tmp:]

        #tratamiento_XGB_season(train_condition,test_condition,opt,'TARGET')
        #tratamiento_SGDseason(train_condition,test_condition,opt,'TARGET')
        #tratamiento_SGDPARTIALseason(train_condition, test_condition, opt, 'TARGET')

        predictionSGD, SGDFITdrifts, SGDPartialdrifts = tratamiento_SGD_WHOLE(X_train, y_train1, X_test, y_test1, opt,"TARGET")
        print('Drift detectados por SGD: ', datasettest.index[SGDFITdrifts])
        print('Drift detectados por Partial SGD: ', datasettest.index[SGDPartialdrifts])
        pred_LIN_VEL.append(predictionSGD[1])

        VISUALIZACION_MEJORA(X_test,y_test1,pred_LIN_VEL[0],opt,"TARGET",'Trajectory Visualization for :')
        cont+=1

    print('FIN')

