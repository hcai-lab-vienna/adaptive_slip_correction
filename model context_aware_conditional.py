import pandas as pd

pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np

from scipy.signal import savgol_filter
import os
from pathlib import Path
from datetime import datetime
from evo.tools import file_interface
from analisis_features_utils import (load_meta,load_trajectory_data2,
                                     treatment_XGB_season,IMPROVEMENT_VISUALIZATION_error,
                                     experiment_RQ1,experiment_RQ2,experiment_RQ3,
                                     ONE_modelelastic,model_Slip_index,conditional_model,
                                     model_one_Slip_index,merge_predictions,conditional_1model,
                                     Save_results)

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
                          (2,'dry road, not raining'),
                          (3,'wet road, raining')]
CONDITION_CLASSIFICATION=[(1,'snow_road'),
                          (2,'dry_road'),
                          (3,'rainy_road')]

FEATURES_METEO = [
    'Rain_accum',
    'RH',
    'SnowDepth_Avg'
]
FEATURES_SV = [
    'lin_acc_imu_x',
    'lin_acc_imu_y',
    'ang_vel_imu_z',
    'grav_x',
    'lin_vel_odom_x',
    'ang_vel_odom_z',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z'
]
FEATURES_L_W_1 = [
    'lin_acc_imu_x',
    'lin_acc_imu_y',
    'ang_vel_imu_z',
    'lin_vel_odom_x',
    'ang_vel_odom_z',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z',
    'grav_x',
    'SV_pred'
]
FEATURES_conditional = [
    'lin_acc_imu_x',
    'lin_acc_imu_y',
    'ang_vel_imu_z',
    'grav_x',
    'lin_vel_odom_x',
    'ang_vel_odom_z',
    'lin_vel_cmd_x',
    'ang_vel_cmd_z',
    'SV_rainy_road_pred',
    'SV_snow_road_pred',
    'SV_dry_road_pred'
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
                no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0)& (df_merged["lin_vel_odom_x"] != 0)]
                start = no_ceros.index[0]
                end = no_ceros.index[-1]
                df_merged = df_merged.loc[start:end]
                df_merged["SV"] = 100 * ((df_merged["TARGET"] - df_merged["lin_vel_cmd_x"]) / df_merged["TARGET"])
                diff = df_merged["SV"].diff().abs()
                threshold = 40
                sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
                big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
                spikes = sign_change & big_jump
                index_spikes = df_merged["SV"][spikes].index
                df_merged["SV"] = df_merged["SV"].clip(-70, 70)
                df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
                mask_bad = ~np.isfinite(df_merged["SV"])
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
            no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0)]
            no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0) & (df_merged["lin_vel_odom_x"] != 0)]
            start = no_ceros.index[0]
            end = no_ceros.index[-1]
            df_merged = df_merged.loc[start:end]
            df_merged["SV"] = 100 * ((df_merged["TARGET"] - df_merged["lin_vel_cmd_x"]) / df_merged[ "TARGET"])
            diff = df_merged["SV"].diff().abs()
            threshold = 40
            sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
            big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
            spikes = sign_change & big_jump
            index_spikes = df_merged["SV"][spikes].index
            df_merged["SV"] = df_merged["SV"].clip(-70, 70)
            df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
            mask_bad = ~np.isfinite(df_merged["SV"])
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
            no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0)]
            no_ceros = df_merged[(df_merged["lin_vel_cmd_x"] != 0) & (df_merged["TARGET"] != 0) & (df_merged["lin_vel_odom_x"] != 0)]
            start = no_ceros.index[0]
            end = no_ceros.index[-1]
            df_merged = df_merged.loc[start:end]
            df_merged["SV"] = 100 * ((df_merged["TARGET"] - df_merged["lin_vel_cmd_x"]) / df_merged["TARGET"])
            diff = df_merged["SV"].diff().abs()
            threshold = 40
            sign_change = np.sign(df_merged["SV"]) != np.sign(df_merged["SV"].shift(1))
            big_jump = (df_merged["SV"] - df_merged["SV"].shift(1)).abs() > threshold
            spikes = sign_change & big_jump
            index_spikes = df_merged["SV"][spikes].index
            df_merged["SV"] = df_merged["SV"].clip(-70, 70)
            df_merged["SV"] = savgol_filter(df_merged["SV"].values, 13, 3)
            mask_bad = ~np.isfinite(df_merged["SV"])
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
    df_traj, pos_odom, odom_index, pos_index, trajectory_GT, error = load_trajectory_data2(fecha, color)

    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_odom_position.csv", pos_odom, delimiter=",", fmt='%.18e')
    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_gt_raw_index.csv", pos_index, delimiter=",", fmt='%.18e')
    np.savetxt(NEW_FOMO_PATH + f"{fecha}_{color}_error.csv", error, delimiter=",", fmt='%.18e')
    file_interface.write_tum_trajectory_file(NEW_FOMO_PATH + f"{fecha}_{color}_trajectory.txt", trajectory_GT)

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

    datasettrain = datasettrain.fillna(-1)
    datasettest = datasettest.fillna(-1)

    #PREPROCESS Slip INDEX
    FEATURES1 = FEATURES_METEO + FEATURES_SV
    length_fetauresSV = [len(FEATURES1) ]

    X_train = datasettrain[FEATURES1]
    datasettrain.index = pd.to_datetime(datasettrain.index)
    y_train = datasettrain["SV"]

    X_test = datasettest[FEATURES1]
    datasettest.index = pd.to_datetime(datasettest.index)
    y_test = datasettest["SV"]

    test_condition_ordenado = dict(sorted(test_condition.items()))
    #experiment_RQ1(train_condition, test_condition_ordenado, FEATURES_SV, 'SV')
    #treatment_XGB_season(train_condition, test_condition_ordenado, FEATURES_SV, 'SV')
    #experiment_RQ2(train_condition, test_condition_ordenado, FEATURES1, 'SV')
    #treatment_XGB_season(train_condition, test_condition_ordenado, FEATURES1, 'SV')

    train_1prediction, test_1prediction = model_one_Slip_index(datasettrain, datasettest, FEATURES_SV)
    train_3prediction,test_3prediction=model_Slip_index(train_condition, test_condition_ordenado, FEATURES_SV)

    my_map = {"snow_road": 1, "dry_road": 2, "rainy_road": 3}
    train_3pred_convertido = {my_map[k]: v for k, v in train_3prediction.items()}
    test_3pred_convertido = {my_map[k]: v for k, v in test_3prediction.items()}

    train_final3 = merge_predictions(train_condition, 0,train_3pred_convertido)
    test_final3 = merge_predictions(test_condition_ordenado,0, test_3pred_convertido)

    datasettrain[['SV_pred', 'SV_real']] = train_1prediction[['SV_pred', 'SV_real']]
    datasettest[['SV_pred', 'SV_real']] = test_1prediction[['SV_pred', 'SV_real']]

    FEATURES_OUR = FEATURES_conditional # FEATURES_METEO +
    FEATURES_ONE =FEATURES_L_W_1 # FEATURES_METEO +
    FEATURES_WITHOUT = FEATURES_SV  # FEATURES_METEO +

    resultsVELWITHOUT, predictionVELWITHOUT = ONE_modelelastic(datasettrain[FEATURES_WITHOUT], datasettrain["TARGET"], datasettest[FEATURES_WITHOUT], datasettest["TARGET"],
                                                FEATURES_WITHOUT, "TARGET")
    train_final3.index = pd.to_datetime(train_final3.index)
    train_final3=train_final3.sort_index()
    X_trainvel3 = train_final3[FEATURES_OUR]
    y_trainvel3 = train_final3["TARGET"]

    test_final3.index = pd.to_datetime(test_final3.index)
    test_final3 = test_final3.sort_index()
    X_testvel3 = test_final3[FEATURES_OUR]
    y_testvel3 = test_final3["TARGET"]

    datasettrain.index = pd.to_datetime(datasettrain.index)
    datasettrain = datasettrain.sort_index()
    X_trainvel1 = datasettrain[FEATURES_ONE]
    y_trainvel1 = datasettrain["TARGET"]

    datasettest.index = pd.to_datetime(datasettest.index)
    datasettest = datasettest.sort_index()
    X_testvel1 = datasettest[FEATURES_ONE]
    y_testvel1 = datasettest["TARGET"]

    M1_resultsVEL, M1_predictionVEL = conditional_1model(X_trainvel1, y_trainvel1, X_testvel1, y_testvel1, FEATURES_ONE, "TARGET")
    M3_resultsVEL, M3_predictionVEL = conditional_model(X_trainvel3, y_trainvel3, X_testvel3, y_testvel3, FEATURES_OUR,
                                                        "TARGET")
    Save_results(NEW_FOMO_PATH,X_testvel3, y_testvel3, M3_predictionVEL, M1_resultsVEL,M1_predictionVEL, predictionVELWITHOUT)


    '''IMPROVEMENT_VISUALIZATION_error(X_testvel3, y_testvel3, M3_predictionVEL, '3SV prediction for improving Trajectory Visualization for :')
    IMPROVEMENT_VISUALIZATION_error(X_testvel1, y_testvel1, M1_predictionVEL, '1SV prediction for improvingTrajectory Visualization for :')
    IMPROVEMENT_VISUALIZATION_error(datasettest[FEATURES_WITHOUT], y_testvel1, predictionVELWITHOUT, 'Trajectory Visualization for :')'''

    experiment_RQ3(X_testvel3, y_testvel3, M3_predictionVEL, M1_resultsVEL,M1_predictionVEL, predictionVELWITHOUT)

    print('END')

