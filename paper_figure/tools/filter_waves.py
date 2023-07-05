import logging

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, lfilter
from tslearn.metrics import cdist_soft_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def emg_filter(data, type="lowpass", lowcut=10, highcut=400, fs=1024, order=1):
    if type == "bandpass":
        b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    elif type == "lowpass":
        b, a = _butter_lowpass(lowcut, fs, order=order)
    elif type == "highpass":
        b, a = _butter_highpass(highcut, fs, order=order)

    y = lfilter(b, a, data)
    y = nk.signal_detrend(y, order=0)
    return y


def _butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="bandpass")


def _butter_lowpass(lowcut, fs, order=1):
    return butter(order, lowcut, btype="lowpass", fs=fs)


def _butter_highpass(highcut, sampling_rate, order=4):
    return butter(order, highcut, btype="highpass", fs=sampling_rate)


def wave_validation(id, all_segments, params):
    data = TimeSeriesScalerMeanVariance().fit_transform(all_segments)[:, :, 0]
    all_dtw_dist = cdist_soft_dtw(data) ** 2
    dtw_sum = np.sum(all_dtw_dist, axis=1, keepdims=True)
    dtw_sum = (dtw_sum - dtw_sum.min()) / (dtw_sum.max() - dtw_sum.min())
    fig, ax = plt.subplots(1, 1)

    del_idx = []
    threshold = params["validate_threshold"]
    for i, wave in enumerate(data):
        if dtw_sum[i] < threshold:
            # ax.plot(wave, color="red")
            del_idx.append(i)

        else:
            pass
            # ax.plot(wave,color="c", alpha=0.6)
    # plt.show()
    # plt.close()

    segments = np.delete(all_segments, del_idx, 0)
    segment_size = segments.shape[0]
    logger.debug("{}: {} -> {}".format(id, all_segments.shape[0], segment_size))

    del all_dtw_dist, dtw_sum, del_idx
    return id, segments
