import logging
import warnings
from multiprocessing import Pool, RLock, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from experiment_tools import augumentation, base_test_split, feature_extractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
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
    rfc_max_depth = trial.suggest_int("max_depth", 1, 1000)
    rfc_n_estimators = trial.suggest_int("n_estimators", 1, 100)
    rfc_min_samples_split = trial.suggest_int("min_samples_split", 8, 16)
    rfc_criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
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

        rfc = RandomForestClassifier(
            max_depth=rfc_max_depth,
            n_estimators=rfc_n_estimators,
            min_samples_split=rfc_min_samples_split,
            criterion=rfc_criterion,
        )
        rfc.fit(x_train_kf, y_train_kf)

        y_pred_prob = rfc.predict_proba(x_valid_kf)[:, 1]
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

    # ==========特徴量抽出==========
    best_params = study.best_trial.params
    X_train = feature_extractor(
        X_train,
        myop_thre=best_params["feat_myop"],
        wamp_thre=best_params["feat_wamp"],
        ssc_thre=best_params["feat_ssc"],
        cc_P=best_params["feat_cc"],
    )
    X_test = feature_extractor(
        X_test,
        myop_thre=best_params["feat_myop"],
        wamp_thre=best_params["feat_wamp"],
        ssc_thre=best_params["feat_ssc"],
        cc_P=best_params["feat_cc"],
    )

    # ==========学習==========
    rfc = RandomForestClassifier(
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        min_samples_split=best_params["min_samples_split"],
        criterion=best_params["criterion"],
    )
    rfc.fit(X_train, y_train)

    # ==========予測==========
    result = pd.DataFrame()
    y_pred_prob = rfc.predict_proba(X_test)[:, 1]
    result["y_pred_p"] = y_pred_prob
    result["label"] = y_test
    result["gesture"] = gesture
    result["test_id"] = id
    result["verify_user"] = verify_user

    # ==========ログ出力==========
    logger.info("verify user: {}".format(verify_user))
    logger.info("gesture: {}".format(gesture))
    logger.info("train data shape: {}".format(X_train.shape))
    logger.info("test data shape: {}".format(X_test.shape))
    logger.info("best params: {}".format(best_params))
    logger.info("train accuracy: {:.2f}".format(rfc.score(X_train, y_train)))
    logger.info("test accuracy: {:.2f}".format(rfc.score(X_test, y_test)))

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
            test_pairs.append((dataset, verify_user, gesture, train_test_ratio, id))
            id += 1

    with Pool(cpu_count()) as p:
        imap = p.imap(func=test, iterable=test_pairs)
        result_list = list(tqdm(imap, total=len(test_pairs), leave=False, desc="learning"))
        p.close()

    result = pd.concat(result_list).reset_index(drop=True)
    save_path = "./result/result.csv"
    result.to_csv(save_path, index=False)
