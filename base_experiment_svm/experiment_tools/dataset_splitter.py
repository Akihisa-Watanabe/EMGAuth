import math

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def base_test_split(df: pd.DataFrame, verify_user: int, verify_gesture: str, split_ratio: float):
    # =======================データセット構築用の情報=======================
    df = df.copy()
    all_users = np.unique(df["user"])  # 全ユーザリスト
    all_gestures = np.unique(df["gesture"])  # 全ジェスチャリスト

    # =======================ラベル付=======================
    label = (df["user"] == verify_user).to_numpy() & (df["gesture"] == verify_gesture).to_numpy()
    df["label"] = label

    # =======================テスト用/学習用の正例データを抽出=======================
    df_verify_user = df.query('user == {} & gesture == "{}"'.format(verify_user, verify_gesture))
    df_train_verify_user = df_verify_user.sample(frac=split_ratio)  # 学習用認証者データ
    df_test_verify_user = df_verify_user.drop(index=df_train_verify_user.index)  # テスト用認証者データ

    L_train_verify_user = len(df_train_verify_user)
    L_test_verify_user = len(df_test_verify_user)
    df.drop(df.query("user == {} ".format(verify_user)).index, inplace=True)

    # =======================学習用の負例データを抽出=======================
    g = df.groupby(["gesture", "user"], group_keys=True)
    L_train_each_gesture = math.ceil(
        L_train_verify_user / (len(all_gestures) * (len(all_users) - 1))
    )
    df_train_others = g.apply(
        lambda x: x.sample(n=L_train_each_gesture, random_state=42)
    )  # 学習用認証者以外のデータ
    del_idx = [idx[2] for idx in df_train_others.index]
    df.drop(del_idx, inplace=True)

    # =======================学習用の負例データを抽出=======================
    g = df.groupby(["gesture", "user"], group_keys=True)
    L_test_each_gesture = math.ceil(L_test_verify_user / (len(all_gestures) * (len(all_users) - 1)))
    df_test_others = g.apply(
        lambda x: x.sample(n=L_test_each_gesture, random_state=42)
    )  # 学習用攻撃者データ

    # =======================結合して学習用データを作成=======================
    X_train = pd.concat([df_train_verify_user, df_train_others])
    X_test = pd.concat([df_test_verify_user, df_test_others])
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
