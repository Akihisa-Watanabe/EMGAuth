import argparse
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tools import calc_scores

plt.rcParams["figure.subplot.bottom"] = 0.33
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams["figure.subplot.left"] = 0.10
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["font.size"] = 13
plt.rcParams["figure.figsize"] = (8, 3)


def calc(result):
    all_gestures = np.unique(result["gesture"])
    all_users = np.unique(result["verify_user"])

    accuracy_dict = defaultdict(list)
    eer_dict = defaultdict(list)
    error_rate_dict = defaultdict(list)
    f1_dict = defaultdict(list)
    tpr_dict = defaultdict(list)
    precision_score_dict = defaultdict(list)
    recall_score_dict = defaultdict(list)
    roc_auc_dict = defaultdict(list)

    for verify_user in all_users:
        user_frame = result.query("verify_user == {}".format(verify_user))
        mean_fpr = np.linspace(0, 1, 100)
        range_test = np.unique(user_frame["test_id"])
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
                accuracy_dict[gesture].append(scores["accuracy"])
                error_rate_dict[gesture].append(scores["error_rate"])
                recall_score_dict[gesture].append(scores["recall_score"])
                precision_score_dict[gesture].append(scores["precision_score"])
                f1_dict[gesture].append(scores["f1_score"])
                eer_dict[gesture].append(scores["eer"])
                roc_auc_dict[gesture].append(scores["auc"])

                fpr, tpr, thresholds = roc_curve(y_test, y_pred_p)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tpr_dict[gesture].append(interp_tpr)

    accuracy = dict()
    error_rate = dict()
    recall_score = dict()
    precision_score = dict()
    f1_score = dict()
    eer = dict()
    roc_auc = dict()

    for gesture in all_gestures:
        accuracy[gesture] = np.mean(accuracy_dict[gesture])
        error_rate[gesture] = np.mean(error_rate_dict[gesture])
        recall_score[gesture] = np.mean(recall_score_dict[gesture])
        precision_score[gesture] = np.mean(precision_score_dict[gesture])
        f1_score[gesture] = np.mean(f1_dict[gesture])
        eer[gesture] = np.mean(eer_dict[gesture])
        roc_auc[gesture] = np.mean(roc_auc_dict[gesture])

    return accuracy, error_rate, recall_score, precision_score, f1_score, eer, roc_auc


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(
            "{}".format(round(height, 2)),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 1),  # 1 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,
        )


if __name__ == "__main__":
    res1_path = pathlib.Path(
        "/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/base_experiment_svm/result/result.csv"
    )
    res2_path = pathlib.Path(
        "/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/normal_test_svm/result/result.csv"
    )
    res3_path = pathlib.Path(
        "/Users/watanabeakihisa/Documents/Research/biometric-verification-experiments/leadked_test_svm/result/result.csv"
    )

    res_path = {res1_path: "Baseline", res2_path: "Random test", res3_path: "Imitation test"}

    markers = ["o", "s", "v"]
    colors = [
        "#868E96",
        "#FA5252",
        "#7950F2",
    ]  # ['#495057','#4263EB','#F03E3E']#['#2196F3', '#4CAF50', '#FF9800']
    i = 0
    j = 0
    X = np.array([k for k in range(1, 12)])
    plt.ylim(0.9, 1)
    for path, test_method in res_path.items():
        result = pd.read_csv(path)
        accuracy, error_rate, recall_score, precision_score, f1_score, eer, roc_auc = calc(result)

        myList = roc_auc.items()
        myList = sorted(myList)
        g, y = zip(*myList)

        x = np.array([j for k in range(1, 12)]) + X
        rect = plt.bar(x, y, label=test_method, color=colors[i], width=0.7)
        autolabel(rect)
        # plt.scatter(x,y, marker=markers[i],label = test_method)
        # plt.plot(x, y, marker=markers[i], linestyle = "--", alpha=0.5, markerfacecolor=colors[i])
        i += 1
        j += 0.25
        break

    # plt.grid(axis='x',linestyle='dotted', color='black')
    # x = x - np.array([0.25 for k in range(1,12)])
    label = list(g)
    plt.xticks(x, g, rotation=45)
    plt.xlabel("Gesture", fontsize=18)
    plt.ylabel("AUC", fontsize=18)
    plt.legend(loc="lower right", borderaxespad=0.2, fontsize=13)
    plt.savefig("gestres.png", format="png", dpi=500)
    plt.show()
