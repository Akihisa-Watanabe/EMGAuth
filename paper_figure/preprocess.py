import itertools
import logging
import os
import re
from multiprocessing import Pool, RLock, cpu_count

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import emg_filter, get_segment_pk, load_yaml, visualize
from tqdm import tqdm

plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.top"] = 0.99
plt.rcParams["figure.subplot.left"] = 0.15
plt.rcParams["figure.subplot.right"] = 0.99


def get_segments(args):
    id, raw_signal, params, artifact = args
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
    t = np.arange(0, len(filtered_signal[1300:6300])) / sampling_rate
    plt.plot(t, raw_signal[1300:6300] * 1000, alpha=0.78, label="raw signal", color="#1C7ED6")
    plt.plot(
        t,
        filtered_signal[1300:6300] * 1000,
        alpha=0.7,
        label="filtered and detrended signal",
        color="#FD7E14",
        linewidth=2,
    )
    plt.xlabel("time [s]", fontsize=19)
    plt.ylabel("sEMG [mV]", fontsize=19)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=14)
    plt.savefig("FS.png", format="png", dpi=500)
    # plt.show()
    plt.close()
    # ----1ファイル内に含まれるセグメントを抽出----
    segments = get_segment_pk(
        filtered_signal, segment_window_size, validate_window_size, artifact_label=artifact
    )

    return id, segments


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

    # ------セグメント抽出------
    for i, arg in enumerate(args):
        if "01_FS" in arg[0]:
            id, segments = get_segments(arg)
            print(arg[0])
            visualize(segments)

    # mean_wave = np.mean(segments,axis=0)
    # for wave in segments:
    #     t = np.arange(0, len(wave)) / 1024
    #     plt.plot(t, wave*1000,alpha=0.5,color='blue')
    # plt.plot(t,mean_wave*1000,alpha=1,color='red')
    # plt.show()
