import logging
import warnings
from multiprocessing import Pool, RLock, cpu_count

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from experiment_tools import augumentation, base_test_split, feature_extractor
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("base_test")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("./base_test.log")
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


def test(args):
    dataset, verify_user, gesture, split_ratio, id = args
    X_train, y_train, X_test, y_test = base_test_split(dataset, verify_user, gesture, split_ratio)

    # ==========データ拡張==========
    X_train, y_train = augumentation(X_train, y_train)

    # ==========学習(最適化)==========
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
    best_params = study.best_trial.params

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
    )  # .values

    best_params |= {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        "random_state": 71,
    }
    # ==========学習==========
    dtrain = lgb.Dataset(X_train, label=y_train)
    # d_test = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    # evals_result = {}  # to record eval results for plotting
    gbm = lgb.train(best_params, dtrain)
    # gbm = lgb.train(best_params, dtrain, valid_sets=[dtrain, d_test], verbose_eval=100,
    #   callbacks=[
    #     lgb.log_evaluation(10),
    #     lgb.record_evaluation(evals_result)
    #     ])
    # print('Plotting metrics recorded during training...')
    # ax = lgb.plot_metric(evals_result, metric='binary_logloss')
    # plt.show()

    # print('Plotting feature importances...')
    # ax = lgb.plot_importance(gbm, max_num_features=10)
    # plt.show()

    # ==========テスト==========
    result = pd.DataFrame()
    y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    result["y_pred_p"] = y_pred_prob
    result["label"] = y_test
    result["gesture"] = gesture
    result["test_id"] = id
    result["verify_user"] = verify_user

    # ==========ログ出力==========
    logger.info("train data shape: {}".format(X_train.shape))
    logger.info("test data shape: {}".format(X_test.shape))
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
    id = 0
    result_list = []
    for verify_user in tqdm(all_users):
        for gesture in tqdm(all_gestures, leave=True):
            if verify_user == 4:
                test((dataset, verify_user, gesture, train_test_ratio, id))
            # test_pair = (dataset, verify_user, gesture, train_test_ratio, id)
            # res = test(test_pair)
            # result_list.append(res)
            id += 1

    result = pd.concat(result_list).reset_index(drop=True)
    save_path = "./result/result.csv"
    result.to_csv(save_path, index=False)
