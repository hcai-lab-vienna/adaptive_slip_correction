import pandas as pd

from BOKU.analisis_features_utils import treatment_TKAN_WHOLE

pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from pathlib import Path
from datetime import datetime
from analisis_features_utils import (treatment_XGB_WHOLE,treatment_SGD_WHOLE,treatment_TKAN_WHOLE,
                                     load_meta,load_trajectory_data2,treatment_SGDseason,
                                     treatment_XGB_season,treatment_SGDPARTIALseason,treatment_TKANseason,
                                     IMPROVEMENT_VISUALIZATION,experiment_1CD,experiment_2CD,
                                     THREE_models,ONE_model,VEL_THREE_models)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "..\\..\\..\\OSR\\code\\fomo-dataset"
NEW_FOMO_PATH = "..\\..\\..\\OSR\\code\\New_fomo_DATASET\\"

date_train = {"2025-11-03","2024-11-28",'2025-05-28',"2024-11-21","2025-09-24", "2025-10-14"}
date_test = { "2025-08-20"}
date_SNOW_road_test = {"2024-11-28"}
date_RAIN_road_test = {"2025-11-03"}
CONDITIONS_DATE=[ ("2025-04-15",3),("2025-11-03",3),("2024-11-28",1),("2025-06-26",2),("2025-09-24",2),("2024-11-21",2),("2025-05-28",2),("2025-08-20",2),("2025-10-14",2)]
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
    'ang_vel_imu_z',
    'grav_x',
    'lin_vel_odom_x',
    'ang_vel_odom_z',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z'
]
LABEL_SV='SV'
FEATURES_L_W_1 = [
    'lin_acc_imu_x',
    'lin_acc_imu_y',
    'ang_vel_imu_z',
    'grav_x',
    'lin_vel_odom_x',
    'ang_vel_odom_z',
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
def through_date(ruta_base):
    # Dictionary: DATE -> condition
    date_to_condition = dict(CONDITIONS_DATE)

    train_condition = {}
    test_condition = {}

    train_final = {}
    test_final = {}

    for cond_id in set(date_to_condition.values()):
        train_condition[cond_id] = []
        test_condition[cond_id] = []

    for fecha_dir in os.listdir(ruta_base):
        ruta_fecha = Path(ruta_base) / fecha_dir
        if not ruta_fecha.is_dir():
            continue
        season = get_season(fecha_dir)
        print(f"\n {fecha_dir} → {season}")

        es_train = fecha_dir in date_train
        es_test = fecha_dir in date_test

        cond_id = date_to_condition.get(fecha_dir)

        for subdir in ruta_fecha.iterdir():
            if subdir.is_dir() and subdir.name.startswith("red"):
                df_meteo, df_imu = analize_directory(subdir,season,fecha_dir)
                #MERGE
                if  df_meteo is None or df_meteo.empty:
                    print("⚠️ EMPTY Meteo.")
                    df_merged = df_imu.copy()
                else:
                    df_imurounded = df_imu.index.round('min')
                    df_meteorounded = df_meteo.index.round('min')

                    hay_interseccion = (
                            df_imurounded.min() <= df_meteorounded.max() and
                            df_meteorounded.min() <= df_imurounded.max()
                    )

                    print("There is intersection:", hay_interseccion)
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
                    print("⚠️ There is NaN:", hay_nan)

                    if hay_nan:
                        print("NaN by column:")
                        print(df_merged.isna().sum())

                # ---------- SLIP INDEX ----------
                no_ceros = df_merged[(df_merged["lin_vel_odom_x"] != 0)&(df_merged["TARGET"] != 0)]
                no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["lin_vel_odom_x"] != 0)]
                no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0)& (df_merged["lin_vel_odom_x"] != 0)]
                start = no_ceros.index[0]
                end = no_ceros.index[-1]
                df_merged = df_merged.loc[start:end]
                df_merged["SV"] = 100 * (
                            (df_merged["TARGET"] - df_merged["lin_vel_odom_x"]) / df_merged[
                        "TARGET"])#JB
                df_merged["SV"] = 100 * (
                        (df_merged["TARGET"] - df_merged["lin_vel_cmd_x"]) / df_merged[
                    "TARGET"])#JB
                diff = df_merged["SV"].diff().abs()
                threshold = 40
                sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
                big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
                spikes = sign_change & big_jump
                index_spikes = df_merged["SV"][spikes].index
                df_merged["SV"] = df_merged["SV"].clip(-70, 70)
                df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
                mask_bad = ~np.isfinite(df_merged['SV'])
                df_merged = df_merged[~mask_bad].copy()

                if es_train:
                    if hay_nan == False:
                        train_condition[cond_id].append(df_merged)
                        print('TRAIN No nan',fecha_dir)
                if es_test:
                    if hay_nan == False:
                        test_condition[cond_id].append(df_merged)
                        print('TEST No nan', fecha_dir)

    #extratrajectory for snow_road for test
    ruta=Path(ruta_base) /'2024-11-28'
    for subdir in ruta.iterdir():
        if subdir.is_dir() and subdir.name.startswith("yellow"):#yellow bue
            df_meteo, df_imu = analize_directory(subdir, 'autumn', '2024-11-28',color="yellow")#yellow
            # MERGE
            if df_meteo is None or df_meteo.empty:
                print("⚠️ Meteo EMPTY")
                df_merged = df_imu.copy()
            else:
                df_imurounded = df_imu.index.round('min')
                df_meteorounded = df_meteo.index.round('min')

                hay_interseccion = (
                        df_imurounded.min() <= df_meteorounded.max() and
                        df_meteorounded.min() <= df_imurounded.max()
                )
                print("THere is intersection:", hay_interseccion)
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
                print("⚠️ There is NaN:", hay_nan)

                if hay_nan:
                    print("NaN by column:")
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
            diff = df_merged["SV"].diff().abs()
            threshold = 40
            sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
            big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
            spikes = sign_change & big_jump
            index_spikes = df_merged["SV"][spikes].index
            df_merged["SV"] = df_merged["SV"].clip(-70, 70)
            df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
            mask_bad = ~np.isfinite(df_merged['SV'])
            df_merged = df_merged[~mask_bad].copy()

            if hay_nan==False:
                test_condition[1].append(df_merged)#SNOW
                print('yellow','2024-11-28')
    # extratrajectory for RAIN for test
    ruta = Path(ruta_base) /'2025-11-03'
    for subdir in ruta.iterdir():
        if subdir.is_dir() and subdir.name.startswith("blue"):
            df_meteo, df_imu = analize_directory(subdir, 'autumn', '2025-11-03', color="blue")
            # MERGE
            if df_meteo is None or df_meteo.empty:
                print("⚠️ Meteo EMPTY.")
                df_merged = df_imu.copy()
            else:
                df_imurounded = df_imu.index.round('min')
                df_meteorounded = df_meteo.index.round('min')

                hay_interseccion = (
                        df_imurounded.min() <= df_meteorounded.max() and
                        df_meteorounded.min() <= df_imurounded.max()
                )
                print("There is intersection:", hay_interseccion)
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
                print("⚠️ There is  NaN:", hay_nan)

                if hay_nan:
                    print("NaN by column:")
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
            diff = df_merged["SV"].diff().abs()
            threshold = 40
            sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
            big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
            spikes = sign_change & big_jump
            index_spikes = df_merged["SV"][spikes].index
            df_merged["SV"] = df_merged["SV"].clip(-70, 70)
            df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
            mask_bad = ~np.isfinite(df_merged['SV'])
            df_merged = df_merged[~mask_bad].copy()

            if hay_nan == False:
                test_condition[3].append(df_merged)  # RAINING
                print('blue', '2025-11-03')

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
def analize_directory(ruta_red,season,fecha,color="red"):
    print("Loading data...")
    df_traj, pos_odom,odom_index,pos_index,pos_gt,error = load_trajectory_data2(fecha,color)

    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_odom_position.csv", pos_odom, delimiter=",", fmt='%.18e')
    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_gt_position.csv", pos_gt,delimiter=",", fmt='%.18e')
    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_gt_raw_index.csv", pos_index, delimiter=",", fmt='%.18e')
    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_error.csv", error, delimiter=",", fmt='%.18e')

    pos_index_col = odom_index.values.astype('datetime64[s]').astype('int64').reshape(-1,1)
    pos_odom_arr = pos_odom if isinstance(pos_odom, np.ndarray) else np.array(pos_odom)

    ceros = np.zeros((pos_index_col.shape[0], 3))
    unos = np.ones((pos_index_col.shape[0], 1))

    if pos_index_col.shape[0]!=pos_odom[1:].shape[0]:
        print('A')

    data = np.hstack([
        pos_index_col,
        pos_odom[1:],  # (N,3)
        ceros,  # (N,3)
        unos  # (N,1)
    ])

    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_output.txt", data, delimiter=" ", fmt='%.18e')

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
        print("The file exist")
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
        # BUILD test
        for cond_id, cond_name in CONDITION_CLASSIFICATION:
            file_path = os.path.join(NEW_FOMO_PATH, f"{cond_name}test.csv")
            test_condition[cond_id] = pd.read_csv(file_path, index_col=0)
    else:
        print("The file doesn't exist")
        train_condition,test_condition,datasettrain, datasettest = through_date(BASE_PATH)
        datasettrain.to_csv(NEW_FOMO_PATH + "DATASET_TRAIN.csv", index=True)
        datasettest.to_csv(NEW_FOMO_PATH + "DATASET_TEST.csv", index=True)

        all_conds = sorted(set(train_condition.keys()) | set(test_condition.keys()))

        train_condition_ordenado = dict(sorted(train_condition.items()))
        test_condition_ordenado = dict(sorted(test_condition.items()))

        for cond in all_conds:
            # Train
            if cond in train_condition_ordenado:
                df_tr = train_condition_ordenado[cond]
                df_tr.to_csv(
                    NEW_FOMO_PATH + f"{CONDITION_CLASSIFICATION[cond - 1][1]}train.csv",
                    index=True
                )

            # Test
            if cond in test_condition_ordenado:
                df_est = test_condition_ordenado[cond]
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
    length_fetauresSV = [len(FEATURES1) ]

    X_train = datasettrain[FEATURES1]
    datasettrain.index = pd.to_datetime(datasettrain.index)
    y_train = datasettrain["SV"]

    X_test = datasettest[FEATURES1]
    datasettest.index = pd.to_datetime(datasettest.index)
    y_test = datasettest["SV"]

    test_condition_ordenado = dict(sorted(test_condition.items()))
    #experiment_1CD(X_train, y_train, test_condition_ordenado, FEATURES1, 'SV')
    #experiment_2CD(train_condition, X_test,y_test, FEATURES1, 'SV')

    #treatment_XGB_season(train_condition,test_condition,FEATURES1,'SV')
    #treatment_SGDseason(train_condition,test_condition,FEATURES1,'SV')
    #treatment_SGDPARTIALseason(train_condition, test_condition, FEATURES1, 'SV')

    '''predictionSGD, SGDFITdrifts, SGDPartialdrifts = treatment_SGD_WHOLE(X_train, y_train, X_test, y_test, FEATURES1,"SV")
    print('Drifts detected by means of SGD: ', datasettest.index[SGDFITdrifts])
    print('Drifts detected by means of Partial SGD: ', datasettest.index[SGDPartialdrifts])
    pred_SV.append(predictionSGD[1])
    '''
    M3_results,M3_prediction,M3model_category,M3_drifts,M3_drifts_date=THREE_models(train_condition, X_test, y_test, FEATURES1,"SV")
    M1_results, M1_prediction= ONE_model(X_train, y_train, X_test, y_test, FEATURES1, "SV")

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


    FEATURES2 = FEATURES_METEO + FEATURES_L_W_1
    # MODELS TO USING SV

    X_train = datasettrain[FEATURES2]
    y_train1 = datasettrain["TARGET"]

    datasettest['SV'][length_fetauresSV[0]:]=M3_prediction
    X_test = datasettest[FEATURES2]
    y_test1 = datasettest["TARGET"]

    tmp=0
    #test_condition_ordenado = dict(sorted(test_condition.items()))
    for key, df in test_condition_ordenado.items():
        if key==1:
            df.loc[df.index[length_fetauresSV[0]:], 'SV'] = M3_prediction[:df.loc[df.index[length_fetauresSV[0]:], 'SV'].shape[0]]
            tmp=df.loc[df.index[length_fetauresSV[0]:], 'SV'].shape[0]
        elif key==2:
            df.loc[df.index[:], 'SV'] = M3_prediction[tmp:df.shape[0]+tmp]
            tmp+= df.shape[0]
        else:
            df.loc[df.index[:], 'SV'] = M3_prediction[tmp:]

    #treatment_XGB_season(train_condition,test_condition,opt,'TARGET')
    #treatment_SGDseason(train_condition,test_condition,opt,'TARGET')
    #treatment_SGDPARTIALseason(train_condition, test_condition, opt, 'TARGET')

    '''predictionSGD, SGDFITdrifts, SGDPartialdrifts = treatment_SGD_WHOLE(X_train, y_train1, X_test, y_test1, FEATURES2,"TARGET")
    print('Drifts detected by means of SGD: ', datasettest.index[SGDFITdrifts])
    print('Drifts detected by means of Partial SGD: ', datasettest.index[SGDPartialdrifts])
    pred_LIN_VEL.append(predictionSGD[1])'''

    M3_resultsVEL, M3_predictionVEL= VEL_THREE_models(train_condition, X_test, y_test1, FEATURES2,"TARGET",M3model_category,M3_drifts,)
    flat_vel = np.concatenate(M3_predictionVEL)
    IMPROVEMENT_VISUALIZATION(X_test, y_test1, flat_vel, FEATURES2, "TARGET", 'Trajectory Visualization for :')

    datasettest['SV'][length_fetauresSV[0]:] = M1_prediction
    X_test = datasettest[FEATURES2]
    y_test2 = datasettest["TARGET"]

    M1_resultsVEL, M1_predictionVEL = ONE_model(X_train, y_train, X_test, y_test2, FEATURES2,"TARGET")

    print('END')

