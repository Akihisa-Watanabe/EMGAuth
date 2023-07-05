import logging

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks
from tslearn.metrics import cdist_soft_dtw

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def get_segment_pk(data, segment_size, validate_size=1000, artifact_label=None, raw_data=None):
    amplitude = nk.emg_amplitude(data)  # 振幅を取得
    pk_idx, _ = find_peaks(
        amplitude, height=np.sqrt(np.nan_to_num(np.var(amplitude))), distance=segment_size
    )  # 最大値
    pk_value = [amplitude[idx] for idx in pk_idx]

    if artifact_label is None:
        artifact_label = np.zeros(len(data))

    all_segments = []
    for j, idx in enumerate(pk_idx):
        if idx < segment_size / 2:  # 最初の波形が十分なサイズでない場合は除外
            continue
        elif data.shape[0] - idx < segment_size / 2:  # 最後の波形が十分なサイズでない場合は除外
            continue

        valid_data = artifact_label[int(idx - validate_size / 2) : int(idx + validate_size / 2)]
        if 1 in valid_data:  # 失敗データは除外
            continue

        if raw_data is not None:
            segment = raw_data[int(idx - segment_size / 2) : int(idx + segment_size / 2)]
        else:
            segment = data[int(idx - segment_size / 2) : int(idx + segment_size / 2)]
        all_segments.append(segment)

    return all_segments


def visualize(data: np.ndarray) -> np.ndarray:
    all_dtw_dist = cdist_soft_dtw(data) ** 2
    dtw_sum = np.sum(all_dtw_dist, axis=1, keepdims=True)
    dtw_sum = (dtw_sum - dtw_sum.min()) / (dtw_sum.max() - dtw_sum.min())
    fig, ax = plt.subplots(1, 1)

    del_idx = []
    for i, wave in enumerate(data):
        if dtw_sum[i] < 0.2:
            ax.plot(wave, color="red")
            del_idx.append(i)

        else:
            ax.plot(wave, color="c", alpha=0.6)
    # plt.show()
    plt.close()
    return del_idx
