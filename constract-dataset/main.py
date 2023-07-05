import itertools
import logging
import os
import re
from multiprocessing import Pool, RLock, cpu_count

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import emg_filter, get_segment_pk, load_yaml, wave_validation
from tqdm import tqdm

logger = logging.getLogger("preprocess")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("./preprocess.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s  %(asctime)s  [%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_segments(args):
    key, raw_signal, params, artifact = args
    # -------parameters----------
    AMP_COEF = params["v"] / params["bits"]
    sampling_rate = params["sampling_rate"]
    cutoff_freq = params["cutoff_freq"]
    filter_type = params["filter_type"]
    filter_order = params["filter_order"]
    segment_window_size = sampling_rate * params["window_time"]
    validate_window_size = sampling_rate * params["validate_time"]
    validate_threshold = params["validate_threshold"]

    # ---------filtering-------------
    raw_signal = raw_signal * AMP_COEF - 2.5
    filtered_signal = emg_filter(
        raw_signal, type=filter_type, highcut=cutoff_freq, fs=sampling_rate, order=filter_order
    )

    # ----1ファイル内に含まれるセグメントを抽出----
    segments = get_segment_pk(
        filtered_signal,
        segment_window_size,
        validate_window_size,
        artifact_label=artifact,
        raw_data=raw_signal,
    )

    # log
    logger.info(f"get {len(segments)} segments from {key}")
    return key, segments


if __name__ == "__main__":
    dataset = h5py.File("../dataset/dataset.hdf5")
    all_params = load_yaml("params.yaml")

    segment_log = pd.DataFrame(columns=["gesture", "user", "n"])
    parameter_log = pd.DataFrame(
        all_params["Preprocess"].values(), index=all_params["Preprocess"].keys()
    )

    # ------全セグメント取得------
    args = []
    for gesture in dataset.keys():
        gesture_all_files = dataset[gesture].keys()  # 全部のファイル名を取得　例: [01_0, 01_1, 02_0, 03_0]
        user_files = [
            (k, list(g))
            for k, g in itertools.groupby(
                gesture_all_files, lambda x: re.findall(r"([0-9]+)_", x)[0]
            )
        ]
        # ユーザーごとにファイル名をまとめる　例: [('01', ['01_0', '01_1'])
        #  , ('02', ['02_0']), ('03', ['03_0'])]
        for user, f_name_list in tqdm(user_files, leave=False):
            for f_name in f_name_list:
                query = "{}/{}".format(gesture, f_name)
                data = np.array(dataset[query])
                emg_signal = data[:, 1]

                if user in ["01", "02", "03", "04"]:
                    artifact = None
                else:
                    artifact = data[:, 2]
                key = "{}_{}".format(user, gesture)
                args.append((key, emg_signal, all_params["Preprocess"], artifact))

    del gesture_all_files, user_files, data, emg_signal

    with Pool(cpu_count()) as p:
        imap = p.imap(func=get_segments, iterable=args)
        all_segments = list(tqdm(imap, total=len(args), leave=False, desc="segmentation process"))
        p.close()

    window_size = int(
        all_params["Preprocess"]["sampling_rate"] * all_params["Preprocess"]["window_time"]
    )

    df_columns = ["user", "gesture"] + [i for i in range(window_size)]
    df = pd.DataFrame(columns=df_columns)

    for tag, data in all_segments:
        gesture = re.findall("[0-9]+_(.*)", tag)[0]
        user = re.findall("[0-9]+", tag)[0]
        df_tmp = pd.DataFrame(columns=[i for i in range(window_size)], data=data)
        df_tmp["gesture"] = gesture
        df_tmp["user"] = int(user)
        df = pd.concat([df, df_tmp]).reset_index(drop=True)

    df.drop(df.query("user == 12").index, inplace=True)
    df["label"] = df.groupby(["user", "gesture"]).ngroup()

    # save as csv
    df.to_csv("../dataset/dataset.csv", index=False)
