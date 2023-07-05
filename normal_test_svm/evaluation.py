import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_tools import calc_scores
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve


def roc_plot(tprs, mean_fpr, aucs, user, axes):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    axes.plot(
        mean_fpr,
        mean_tpr,
        label="User: {} (AUC = {:.2f} $\pm$ {:.2f})".format(user, mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )


def main():
    result = pd.read_csv("result/result.csv")

    all_gestures = np.unique(result["gesture"])
    all_users = np.unique(result["verify_user"])
    fig0, ax0 = plt.subplots(tight_layout=True)
    ax0.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    fig1, ax1 = plt.subplots(tight_layout=True)
    data = []
    for verify_user in all_users:
        user_frame = result.query("verify_user == {}".format(verify_user))

        accuracy_list = []
        error_rate_list = []
        recall_score_list = []
        precision_score_list = []
        f1_score_list = []
        eer_list = []
        auc_list = []
        tprs = []

        mean_fpr = np.linspace(0, 1, 100)

        for gesture in all_gestures:
            gesture_frame = user_frame.query('gesture == "{}"'.format(gesture))
            range_test = np.unique(gesture_frame["test_id"])

            for n in range_test:
                test_frame = gesture_frame.query("test_id == {}".format(n))
                y_test = test_frame["test_user"].to_numpy() == test_frame["verify_user"].to_numpy()

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
                tprs.append(interp_tpr)
                if scores["accuracy"] <= 0.5:
                    print(gesture)
                    print("accuracy: {:.2f}".format(scores["accuracy"]))
                    print("recall: ", scores["recall_score"])
                    print("precision: ", scores["precision_score"])
                    print(
                        "verify user vs test user: {} , {}".format(
                            np.unique(test_frame["verify_user"]), np.unique(test_frame["test_user"])
                        )
                    )
                    print("--------------------------")

            # verify_users = np.unique(gesture_frame['verify_user'])
            # test_users = np.unique(gesture_frame['test_user'])
            # test_numbers = np.unique(user_frame['test_id'])

        data.append(accuracy_list)
        roc_plot(tprs, mean_fpr, auc_list, user=verify_user, axes=ax0)
    ax1.boxplot(data)
    ax1.set_xticklabels(all_users)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("user", fontsize=10)  # X軸ラベル
    ax1.set_ylabel("Accuracy", fontsize=10)  # Y軸ラベル
    ax1.grid()
    fig1.savefig("result_accuracy.png", format="png", dpi=500)

    ax0.grid()
    ax0.legend()
    ax0.set_xlabel("FPR", fontsize=10)
    ax0.set_ylabel("TPR", fontsize=10)
    ax0.legend(fontsize=5)
    fig0.savefig("result_roc.png", format="png", dpi=500)
    plt.show()
    print(np.mean(data), np.std(data))


if __name__ == "__main__":
    main()
