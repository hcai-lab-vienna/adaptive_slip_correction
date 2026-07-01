import pandas as pd

pd.set_option('display.float_format', '{:.4f}'.format)
import numpy as np
import matplotlib.pyplot as plt

from analisis_features_utils import (load_trajectory_results,count_outliers)

# ============================================================
# CONFIGURATION
# ============================================================

NEW_FOMO_PATH = "..\\..\\..\\OSR\\code\\New_fomo_DATASET\\"

date_test = { "2025-08-20"}
date_SNOW_road_test = {"2024-11-28"}
date_RAIN_road_test = {"2025-11-03"}


if __name__ == '__main__':
    # ============================================================
    # LOAD RESULTS
    # ============================================================
    df_res = load_trajectory_results(NEW_FOMO_PATH+"comprehensive_analysis_of_trajectories.csv")
    unique_dates=["2024-11-28","2025-08-20","2025-11-03"]
    if df_res is not None:
        for mydate in unique_dates:
            datos_dia = df_res.loc[mydate]

            fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
            titulo = str('Comparison 3 models for the Linear Velocity Prediction ' + str(mydate))
            fig.suptitle(titulo, fontsize=16)
            axs[0].set_title("Trajectories")
            axs[0].plot(datos_dia['gt_x'], datos_dia['gt_y'], label="Recon. GT ", alpha=0.5, color='r', linewidth=1)
            axs[0].plot(datos_dia['predw_x'], datos_dia['predw_y'], label="Recon. Pred. without SV", alpha=0.5,
                        color='green', linewidth=1)
            axs[0].plot(datos_dia['pred3_x'], datos_dia['pred3_y'], label="Recon. Pred. 3SV", alpha=0.9,
                        color='turquoise', linewidth=1)
            axs[0].plot(datos_dia['pred1_x'], datos_dia['pred1_y'], label="Recon. Pred. 1SV", alpha=0.5, color='blue',
                        linewidth=1, linestyle='--')
            axs[0].legend(loc='upper right')
            axs[0].set_xlabel("Coord X (m)")
            axs[0].set_ylabel('Coord Y (m)')
            axs[1].boxplot([datos_dia['rpe_predw'], datos_dia['rpe_pred1'], datos_dia['rpe_pred3']], showfliers=True)
            axs[1].set_title("RPE from relative position")
            axs[1].set_ylabel('RPE (m)')
            axs[1].set_xticklabels(["Model without SV", "1SV Model", "3SV Model"])
            axs[1].tick_params(labelleft=True)
            plt.tight_layout()
            plt.show(block=True)

            reduction3 = 100 * (np.median(datos_dia['rpe_odom']) - np.median(datos_dia['rpe_pred3'])) / np.median(datos_dia['rpe_odom'])
            reduction1 = 100 * (np.median(datos_dia['rpe_odom']) - np.median(datos_dia['rpe_pred1'])) / np.median(datos_dia['rpe_odom'])
            reductionw = 100 * (np.median(datos_dia['rpe_odom']) - np.median(datos_dia['rpe_predw'])) / np.median(datos_dia['rpe_odom'])

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
            media_error = np.mean(datos_dia['rpe_odom'])
            ODOM_abs = np.abs(datos_dia['rpe_odom'])
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

            media_error = np.mean(datos_dia['rpe_pred3'])
            PRED_abs = np.abs(datos_dia['rpe_pred3'])
            mae = np.mean(PRED_abs)
            max2 = np.max(PRED_abs)

            num, out = count_outliers(PRED_abs)
            print('\n 3SV METHOD: ', num, "outliers PRED")

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

            media_error = np.mean(datos_dia['rpe_pred1'])
            PRED_abs = np.abs(datos_dia['rpe_pred1'])
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

            media_error = np.mean(datos_dia['rpe_predw'])
            PRED_abs = np.abs(datos_dia['rpe_predw'])
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
            plt.title(f"LINEAR VELOCITY {mydate}")
            plt.plot(datos_dia['Vel_GT'], label="Ground Truth", alpha=0.5, color='r', linewidth=1)
            plt.plot(datos_dia['inf_w'], label="Without SV Pred.", alpha=0.5, color='green', linewidth=1)
            plt.plot(datos_dia['inf_pred3'], label="3SV Pred.", alpha=0.9, color='turquoise', linewidth=1)
            plt.plot(datos_dia['inf_pred1'], label="1SVPred.", alpha=0.5, color='blue', linewidth=1, linestyle='--')
            plt.legend()
            plt.show(block=True)

    print('END')

