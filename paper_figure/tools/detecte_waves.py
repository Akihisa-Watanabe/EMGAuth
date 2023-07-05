import logging

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_soft_dtw

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def get_segment_pk(data, segment_size, validate_size=1000, artifact_label=None):
    amplitude = nk.emg_amplitude(data)  # 振幅を取得
    pk_idx, _ = find_peaks(
        amplitude, height=np.sqrt(np.nan_to_num(np.var(amplitude))), distance=segment_size
    )  # 最大値
    pk_value = [amplitude[idx] for idx in pk_idx]

    t = np.arange(0, len(amplitude[1300:6300])) / 1024
    _pk_idx = [pkidx for pkidx in pk_idx if (pkidx < 6300) and (pkidx > 1300)]
    graph_pk_value = [amplitude[idx] * 1000 for idx in _pk_idx]
    graph_pk_idx = [(idx - 1300) / 1024 for idx in _pk_idx]

    start_idx = [i - validate_size / 2 for i in _pk_idx]
    end_idx = [i + validate_size / 2 for i in _pk_idx]
    graph_end_idx = [(idx - 1300) / 1024 for idx in end_idx]
    graph_start_idx = [(idx - 1300) / 1024 for idx in start_idx]
    plt.style.context("classic")
    plt.plot(t, amplitude[1300:6300] * 1000, color="#009688", label="sEMG amplitude", alpha=1, lw=2)
    plt.scatter(graph_pk_idx, graph_pk_value, color="#009688", alpha=0.8, label="detected peaks")
    plt.vlines(
        graph_start_idx,
        0,
        max(amplitude[1300:6300] * 1000),
        color="#2196F3",
        alpha=0.8,
        linestyles="dashed",
        lw=2,
    )
    plt.vlines(
        graph_end_idx,
        0,
        max(amplitude[1300:6300] * 1000),
        color="#F44336",
        alpha=0.8,
        linestyles="dashed",
        lw=2,
    )
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=14)
    plt.xlabel("time [s]", fontsize=19)
    # plt.ylabel('sEMG',fontsize=15)
    plt.savefig("amp.png", format="png", dpi=500)
    plt.close()

    if artifact_label is None:
        artifact_label = np.zeros(len(data))

    all_segments = []
    for j, idx in enumerate(pk_idx):
        if idx in _pk_idx:
            print(j)
        if idx < segment_size / 2:  # 最初の波形が十分なサイズでない場合は除外
            continue
        elif data.shape[0] - idx < segment_size / 2:  # 最後の波形が十分なサイズでない場合は除外
            continue

        valid_data = artifact_label[int(idx - validate_size / 2) : int(idx + validate_size / 2)]
        if 1 in valid_data:  # 失敗データは除外
            continue

        segment = data[int(idx - segment_size / 2) : int(idx + segment_size / 2)]
        all_segments.append(segment)

    return all_segments


def visualize(data: list):
    km = TimeSeriesKMeans(
        n_clusters=1, metric="dtw", max_iter=5, max_iter_barycenter=5, random_state=42
    ).fit(np.array(data))

    center = km.cluster_centers_[0].ravel()
    for wave in data:
        t = np.arange(0, len(wave)) / 1024
        plt.plot(t, wave * 1000, alpha=0.7, color="#B0BEC5")
    plt.plot(t, center * 1000, color="black", linestyle="-", linewidth=2, label="cluster center")
    plt.xlabel("time [s]", fontsize=19)
    plt.ylabel("sEMG [mV]", fontsize=19)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=14)
    plt.savefig("clusters.png", format="png", dpi=500)
    plt.show()

    # all_dtw_dist = cdist_soft_dtw(data) ** 2
    # dtw_sum = np.sum(all_dtw_dist, axis=1, keepdims=True)
    # dtw_sum  = (dtw_sum - dtw_sum.min()) / (dtw_sum.max() - dtw_sum.min())
    # fig, ax = plt.subplots(1, 1)

    # del_idx = []
    # for i,wave in enumerate(data):
    #     ax.plot(wave, color="red")
    #     if dtw_sum[i] <0.2:
    #         # ax.plot(wave, color="red")
    #         del_idx.append(i)

    #     else:
    #         # ax.plot(wave,c
    #         # olor="c", alpha=0.6)
    # plt.show()
    # plt.close()
    # return del_idx
