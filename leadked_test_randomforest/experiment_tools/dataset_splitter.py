import math
import random

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def leaked_test_split(
    df: pd.DataFrame, verify_user: int, verify_gesture: str, attacker: int, split_ratio: float
):
    # =======================データセット構築用の情報=======================
    df = df.copy()
    all_users = np.unique(df["user"])  # 全ユーザリスト
    all_gestures = np.unique(df["gesture"])  # 全ジェスチャリスト
    C_gestures = list(set(all_gestures) - set([verify_gesture]))
    train_attackers = list(set(all_users) - set([verify_user, attacker]))

    # =======================ラベル付=======================
    label = (df["user"] == verify_user).to_numpy() & (df["gesture"] == verify_gesture).to_numpy()
    df["label"] = label

    # =======================テスト用/学習用の正例データを抽出=======================
    df_verify_user = df.query('user == {} & gesture == "{}"'.format(verify_user, verify_gesture))
    df_train_verify_user = df_verify_user.sample(frac=split_ratio)  # 学習用認証者データ
    df_test_verify_user = df_verify_user.drop(index=df_train_verify_user.index)  # テスト用認証者データ
    L_train_verify_user = len(df_train_verify_user)
    L_test_verify_user = len(df_test_verify_user)

    df.drop(df.query("user == {}".format(verify_user)).index, inplace=True)

    # =======================テスト用の負例データを抽出=======================
    df_attacker = df.query('user=={} & gesture == "{}"'.format(attacker, verify_gesture))
    if len(df_attacker) > L_test_verify_user:
        df_test_attacker = df_attacker.sample(n=L_test_verify_user)
    else:
        df_test_attacker = df_attacker
    # テスト用攻撃者データ
    df.drop(df.query("user ==  {}".format(attacker)).index, inplace=True)

    # =======================学習用の負例データを抽出=======================
    df_trainers = df.query("user in {}".format(train_attackers))
    g = df_trainers.groupby(["gesture", "user"], group_keys=True)
    L_train_each_gesture = math.ceil(L_train_verify_user / (len(C_gestures) * len(train_attackers)))
    df_train_attackers = g.apply(
        lambda x: x.sample(n=L_train_each_gesture, random_state=42)
    )  # 学習用攻撃者データ

    df.drop(df.query("user in {}".format(train_attackers)).index, inplace=True)

    # =======================結合して学習用/テスト用データを作成=======================
    X_train = pd.concat([df_train_verify_user, df_train_attackers])
    X_test = pd.concat([df_test_verify_user, df_test_attacker])
    y_train = X_train.pop("label")
    y_test = X_test.pop("label")

    # =======================アンダーサンプリング=======================
    under = RandomUnderSampler(random_state=100, sampling_strategy=1.0)
    X_train, y_train = under.fit_resample(X_train, y_train)
    X_test, y_test = under.fit_resample(X_test, y_test)

    # =======================余計な列を削除=======================
    delete_columns = ["user", "gesture"]

    X_train.drop(columns=delete_columns, inplace=True)
    X_test.drop(columns=delete_columns, inplace=True)

    return X_train, y_train, X_test, y_test
