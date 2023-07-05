import logging
import warnings
from multiprocessing import Pool, RLock, cpu_count

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from experiment_tools import augumentation, feature_extractor, normal_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("normal_test")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("./normal_test.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s  %(asctime)s  [%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def objective(trial, x_train, y_train, n_splits=5):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 300, 700),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 2),
        "max_depth": trial.suggest_int("max_depth", 2, 100),
        "path_smooth": trial.suggest_float("path_smooth", 0, 10),
        "verbose": -1,
        "random_state": 71,
        "n_jobs": cpu_count(),
    }
    feat_myop = trial.suggest_float("feat_myop", 1e-7, 0.01)
    feat_wamp = trial.suggest_float("feat_wamp", 1e-7, 0.01)
    feat_ssc = trial.suggest_float("feat_ssc", 1e-7, 0.0001)
    feat_cc = trial.suggest_int("feat_cc", 1, 10)

    x_train = feature_extractor(
        x_train, myop_thre=feat_myop, wamp_thre=feat_wamp, ssc_thre=feat_ssc, cc_P=feat_cc
    ).values

    kf = StratifiedKFold(n_splits, shuffle=True, random_state=71)
    scores = []
    for tr_idx, val_idx in kf.split(x_train, y_train):
        x_train_kf, x_valid_kf = x_train[tr_idx], x_train[val_idx]
        y_train_kf, y_valid_kf = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(x_train_kf, label=y_train_kf)
        gbm = lgb.train(params, dtrain)
        y_pred_prob = gbm.predict(x_valid_kf, num_iteration=gbm.best_iteration)
        logloss = log_loss(y_valid_kf, y_pred_prob)
        scores.append(logloss)

    return -1 * np.mean(scores)


def train_and_test(args):
    dataset, verify_user, gesture, attacker, train_test_ratio, id = args
    X_train, y_train, X_test, y_test = normal_test_split(
        dataset, verify_user, gesture, attacker, train_test_ratio
    )

    # ==========データ拡張==========
    X_train, y_train = augumentation(X_train, y_train)

    # ==========パラメタ最適化==========
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()
    # fig = optuna.visualization.plot_slice(study, params=["lambda_l1", "lambda_l2"])
    # fig.show()

    best_params = study.best_params
    best_params |= {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        "random_state": 71,
        "n_jobs": -1,
    }

    # ==========ログ出力==========
    logger.info("verify user: {}".format(verify_user))
    logger.info("gesture: {}".format(gesture))
    logger.info("best params: {}".format(best_params))

    # ==========特徴量抽出==========
    bp_feat_myop = best_params.pop("feat_myop")
    bp_feat_wamp = best_params.pop("feat_wamp")
    bp_feat_ssc = best_params.pop("feat_ssc")
    bp_feat_cc = best_params.pop("feat_cc")

    X_train = feature_extractor(
        X_train,
        myop_thre=bp_feat_myop,
        wamp_thre=bp_feat_wamp,
        ssc_thre=bp_feat_ssc,
        cc_P=bp_feat_cc,
    ).values
    X_test = feature_extractor(
        X_test,
        myop_thre=bp_feat_myop,
        wamp_thre=bp_feat_wamp,
        ssc_thre=bp_feat_ssc,
        cc_P=bp_feat_cc,
    ).values

    dtrain = lgb.Dataset(X_train, label=y_train)

    gbm = lgb.train(best_params, dtrain)

    # =============評価===============
    result = pd.DataFrame()
    y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    result["test_user"] = np.where(y_test == 1, verify_user, attacker)
    result["verify_user"] = verify_user
    result["y_pred_p"] = y_pred_prob
    result["gesture"] = gesture
    result["test_id"] = id

    # ==========ログ出力==========
    logger.info("attacker: {}".format(attacker))
    logger.info("train data shape: {}".format(X_train.shape))
    logger.info("test data shape: {}".format(X_test.shape))
    logger.info("best params: {}".format(best_params))
    logger.info(
        "train accuracy: {:.2f}".format(
            accuracy_score(y_train, gbm.predict(X_train, num_iteration=gbm.best_iteration) > 0.5)
        )
    )
    logger.info("test accuracy: {:.2f}".format(accuracy_score(y_test, y_pred_prob > 0.5)))
    return result


if __name__ == "__main__":
    dataset = pd.read_csv("../dataset/dataset.csv")
    all_users = np.unique(dataset["user"])
    all_gestures = np.unique(dataset["gesture"])

    train_test_ratio = 0.5

    test_pairs = []
    id = 0
    for verify_user in all_users:
        for gesture in all_gestures:
            attackers = list(set(all_users) - set([verify_user]))
            for attacker in attackers:
                test_pairs.append((dataset, verify_user, gesture, attacker, train_test_ratio, id))
                id += 1

    result_list = []
    for test_pair in tqdm(test_pairs, desc="learning"):
        result_list.append(train_and_test(test_pair))

    result = pd.concat(result_list).reset_index(drop=True)
    save_path = "./result/result.csv"
    result.to_csv(save_path, index=False)
