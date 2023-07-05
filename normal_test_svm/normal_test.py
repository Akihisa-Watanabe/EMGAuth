import logging
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import optuna
import pandas as pd
from experiment_tools import augumentation, feature_extractor, normal_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
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


def objective(trial, X_train, y_train, n_splits=5):
    svc_c = trial.suggest_float("svc_c", 1e-10, 1e3, log=True)
    svc_gamma = trial.suggest_float("svc_gamma", 1e-10, 1e3, log=True)
    feat_myop = trial.suggest_float("feat_myop", 1e-7, 0.01)
    feat_wamp = trial.suggest_float("feat_wamp", 1e-7, 0.01)
    feat_ssc = trial.suggest_float("feat_ssc", 1e-7, 0.0001)
    feat_cc = trial.suggest_int("feat_cc", 1, 10)

    x_train = feature_extractor(
        X_train, myop_thre=feat_myop, wamp_thre=feat_wamp, ssc_thre=feat_ssc, cc_P=feat_cc
    ).values

    scaler = MinMaxScaler()
    kf = StratifiedKFold(n_splits, shuffle=True, random_state=71)
    scores = []

    for tr_idx, val_idx in kf.split(x_train, y_train):
        x_train_kf, x_valid_kf = x_train[tr_idx], x_train[val_idx]
        y_train_kf, y_valid_kf = y_train[tr_idx], y_train[val_idx]

        x_train_kf = scaler.fit_transform(x_train_kf)
        x_valid_kf = scaler.transform(x_valid_kf)
        svc = SVC(probability=True, max_iter=5000, C=svc_c, gamma=svc_gamma)
        svc.fit(x_train_kf, y_train_kf)

        y_pred_prob = svc.predict_proba(x_valid_kf)[:, 1]
        logloss = log_loss(y_valid_kf, y_pred_prob)
        scores.append(logloss)

    return -1 * logloss


def train_and_test(args):
    dataset, verify_user, gesture, attacker, train_test_ratio, id = args
    X_train, y_train, X_test, y_test = normal_test_split(
        dataset, verify_user, gesture, attacker, train_test_ratio
    )

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

    # ==========正規化==========
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==========学習==========
    svm = SVC(
        probability=True, max_iter=5000, C=best_params["svc_c"], gamma=best_params["svc_gamma"]
    )
    svm.fit(X_train, y_train)

    # ==========ログ出力==========
    logger.info("verify user: {}".format(verify_user))
    logger.info("gesture: {}".format(gesture))
    logger.info("train data shape: {}".format(X_train.shape))
    logger.info("test data shape: {}".format(X_test.shape))
    logger.info("best params: {}".format(best_params))
    logger.info("train accuracy: {:.2f}".format(svm.score(X_train, y_train)))
    logger.info("test accuracy: {:.2f}".format(svm.score(X_test, y_test)))

    # =============評価===============
    result = pd.DataFrame()
    y_pred_prob = svm.predict_proba(X_test)[:, 1]
    result["test_user"] = np.where(y_test == 1, verify_user, attacker)
    result["verify_user"] = verify_user
    result["y_pred_p"] = y_pred_prob
    result["gesture"] = gesture
    result["test_id"] = id
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

    with Pool(cpu_count()) as p:
        imap = p.imap(func=train_and_test, iterable=test_pairs)
        result_list = list(tqdm(imap, total=len(test_pairs), leave=False, desc="learning"))
        p.close()

    result = pd.concat(result_list).reset_index(drop=True)
    save_path = "./result/result.csv"
    result.to_csv(save_path, index=False)
