import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from tools import calc_scores


def calc(result):
    all_gestures = np.unique(result["gesture"])
    all_users = np.unique(result["verify_user"])

    accuracys = []
    eers = []
    f1s = []
    tprs = []

    for verify_user in all_users:
        user_frame = result.query("verify_user == {}".format(verify_user))

        accuracy_list = []
        error_rate_list = []
        recall_score_list = []
        precision_score_list = []
        f1_score_list = []
        eer_list = []
        auc_list = []
        tprs_list = []

        mean_fpr = np.linspace(0, 1, 100)

        for gesture in all_gestures:
            gesture_frame = user_frame.query('gesture == "{}"'.format(gesture))
            range_test = np.unique(gesture_frame["test_id"])

            for n in range_test:
                test_frame = gesture_frame.query("test_id == {}".format(n))
                try:
                    y_test = test_frame["label"].to_numpy()
                except KeyError:
                    y_test = (
                        test_frame["test_user"].to_numpy() == test_frame["verify_user"].to_numpy()
                    )

                y_pred_p = test_frame["y_pred_p"]
                y_pred = (y_pred_p > 0.5).astype(int)

                scores = calc_scores(y_pred, y_pred_p, y_test)
                accuracy_list.append(scores["accuracy"])
                error_rate_list.append(scores["error_rate"])
                recall_score_list.append(scores["recall_score"])
                precision_score_list.append(scores["precision_score"])
                f1_score_list.append(scores["f1_score"])
                eer_list.append(scores["eer"])
                auc_list.append(scores["auc"])

                fpr, tpr, thresholds = roc_curve(y_test, y_pred_p)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_list.append(interp_tpr)

        tprs.append(np.mean(tprs_list, axis=0))

    return np.mean(tprs, axis=0)


def roc_plot(tprs, axes, model):
    mean_fpr = np.linspace(0, 1, 100)
    tprs[-1] = 1.0
    mean_auc = auc(mean_fpr, tprs)

    axes.plot(
        mean_fpr,
        tprs,
        label="{} (Mean AUC: {})".format(model, round(mean_auc, 2)),
        lw=2,
        alpha=0.8,
    )


if __name__ == "__main__":
    # res1_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/normal_test_svm/result/result.csv")
    # res2_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/normal_test_lightbgm/result/result.csv")
    # res3_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/normal_test_randomforest/result/result.csv")
    res1_path = pathlib.Path(
        "/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/base_experiment_svm/result/result.csv"
    )
    res2_path = pathlib.Path(
        "/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/base_experiment_triplet/result/result.csv"
    )
    res3_path = pathlib.Path(
        "/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/base_experiment_randomforest/result/result.csv"
    )
    # res4_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/base_experiment_triplet/result/result.csv")
    # res1_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/leadked_test_svm/result/result.csv")
    # res2_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/leadked_test_lightbgm/result/result.csv")
    # res3_path = pathlib.Path("/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/leadked_test_randomforest/result/result.csv")
    res_path = {res1_path: "SVM", res2_path: "1dConvTriplet", res3_path: "RandomForest"}

    fig0, ax0 = plt.subplots(tight_layout=True)
    roc_x = np.linspace(0, 1, 100)
    eer_line = roc_x[::-1]
    ax0.plot(roc_x, eer_line, linestyle="--", lw=1.5, color="black", alpha=0.7)

    tprs = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    eers = []
    i = 0
    for path, model in res_path.items():
        result = pd.read_csv(path)
        mean_tpr = calc(result)
        tprs.append(mean_tpr)
        eer_idx = np.argmin(np.abs(1 - mean_tpr - roc_x))

        ### normal test ###
        # eer=roc_x[eer_idx]
        # ax0.vlines(roc_x[eer_idx], 0, mean_tpr[eer_idx], linestyles="--", colors=colors[i], lw=2)

        ### baseline ###
        if i == 0:
            ax0.vlines(
                roc_x[eer_idx], 0, mean_tpr[eer_idx], linestyles="--", colors=colors[i], lw=2
            )

        elif i == 1:
            line = Line2D(
                [roc_x[eer_idx], roc_x[eer_idx]],
                [0, mean_tpr[eer_idx]],
                linestyle="--",
                color=colors[i],
                lw=2,
            )
            line.set_dashes([3.7, 1.6])
            ax0.add_artist(line)
        elif i == 2:
            line = Line2D(
                [roc_x[eer_idx], roc_x[eer_idx]],
                [0, mean_tpr[eer_idx]],
                linestyle="--",
                color=colors[i],
                lw=2,
            )
            line.set_dashes([3.7, 1.6 * 2 + 3.7])
            ax0.add_artist(line)

        ### leaked test ###
        # if i==2:
        #     ax0.vlines(roc_x[eer_idx], 0, mean_tpr[eer_idx], linestyles="--", colors=colors[i],lw=2)
        # elif i==0:
        #     line = Line2D([roc_x[eer_idx],roc_x[eer_idx]], [0, mean_tpr[eer_idx]], linestyle="--", color=colors[i], lw=2)
        #     line.set_dashes([3.7,1.6])
        #     ax0.add_artist(line)
        # elif i==1:
        #     line = Line2D([roc_x[eer_idx],roc_x[eer_idx]], [0, mean_tpr[eer_idx]], linestyle="--", color=colors[i], lw=2)
        #     line.set_dashes([3.7,1.6*2+3.7])
        #     ax0.add_artist(line)

        ax0.plot(roc_x[eer_idx], mean_tpr[eer_idx], marker="o", color=colors[i], lw=2)

        eers.append(roc_x[eer_idx])
        roc_plot(mean_tpr, ax0, model)
        i += 1
    models = list(reversed(list(res_path.values())))
    for i, eer in enumerate(reversed(eers)):
        print(i, models[i])
        ax0.text(
            0.2,
            0.3 + 0.1 * i,
            "EER({}):{}".format(models[i], round(eer, 3)),
            fontsize=15,
            color=list(reversed(colors))[i],
            backgroundcolor="white",
        )
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel("FPR", fontsize=20)  # baseline->17
    ax0.set_ylabel("TPR", fontsize=20)
    ax0.tick_params(labelsize=15)
    ax0.legend(fontsize=14, loc="lower right", borderaxespad=0.2)
    fig0.savefig("result_roc.png", format="png", dpi=500)
    plt.show()
