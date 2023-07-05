import gc
import math

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


class DatasetSplitter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.all_users = np.unique(self.df["user"])
        self.all_gestures = np.unique(self.df["gesture"])

    def _get_labeled_data(self, verify_user: int, verify_gesture: str) -> pd.DataFrame:
        labeled_data = self.df.copy()
        labeled_data["label"] = (labeled_data["user"] == verify_user) & (
            labeled_data["gesture"] == verify_gesture
        )
        return labeled_data

    def _split_enroll_test(self, df: pd.DataFrame, N: int):
        enroll_data = df.head(N)
        test_data = df.iloc[N:]
        return enroll_data, test_data

    def _get_test_data(self, df: pd.DataFrame, L_test_verify_user: int):
        L_test_each_gesture = math.ceil(
            L_test_verify_user / (len(self.all_gestures) * (len(self.all_users) - 1))
        )

        test_data = df.groupby(["gesture", "user"], group_keys=True).apply(
            lambda x: x.sample(n=L_test_each_gesture, random_state=42)
        )

        return test_data

    def get_train_data(self, verify_user: int, verify_gesture: str):
        labeled_data = self._get_labeled_data(verify_user, verify_gesture)
        df_train = labeled_data.loc[labeled_data["label"] == False]

        g = df_train.groupby(["gesture", "user"], group_keys=True)
        group_to_id = {group: idx for idx, group in enumerate(g.groups)}
        y_train = df_train.apply(lambda row: group_to_id[(row["gesture"], row["user"])], axis=1)
        X_train = g.apply(lambda _: _[:])
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        under_sampler = RandomUnderSampler(random_state=100)
        X_train, y_train = under_sampler.fit_resample(X_train, y_train)
        X_train.drop(columns=["user", "gesture", "label"], inplace=True)

        del labeled_data, df_train
        gc.collect()

        return X_train.values, y_train.values

    def get_test_data(self, verify_user: int, verify_gesture: str, enroll_N: int):
        labeled_data = self._get_labeled_data(verify_user, verify_gesture)
        df_verify_user = labeled_data.query(
            'user == {} & gesture == "{}"'.format(verify_user, verify_gesture)
        )
        X_enroll, df_test_verify_user = self._split_enroll_test(df_verify_user, enroll_N)

        X_enroll.reset_index(drop=True, inplace=True)
        df_test_verify_user.reset_index(drop=True, inplace=True)

        L_test_verify_user = len(df_test_verify_user)
        df_test_others = self._get_test_data(labeled_data, L_test_verify_user)

        # Get negative samples for X_enroll
        df_enroll_negatives = df_test_others.query("label == False").sample(
            n=enroll_N, random_state=42
        )
        df_test_others = df_test_others.drop(df_enroll_negatives.index)

        X_enroll = pd.concat([X_enroll, df_enroll_negatives])
        y_enroll = np.concatenate([np.ones(enroll_N), np.zeros(enroll_N)])

        X_test = pd.concat([df_test_verify_user, df_test_others])
        y_test = X_test.pop("label")
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        X_enroll = X_enroll.drop(columns=["user", "gesture", "label"])
        X_test = X_test.drop(columns=["user", "gesture"])

        under_sampler = RandomUnderSampler(random_state=100)
        X_test, y_test = under_sampler.fit_resample(X_test, y_test)

        del labeled_data, df_verify_user, df_test_verify_user, df_test_others
        gc.collect()

        return X_enroll.values, X_test.values, y_test.values, y_enroll
