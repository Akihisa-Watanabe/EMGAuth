from multiprocessing import Pool

import numpy as np
from scipy.fft import fft
from scipy.signal import lfilter
from scipy.stats import skew

"""
list of functions.

get_skewse
get_IEMG
get_MAV
get_MAV1
get_MAV2
get_SSI
get_VAR
get_TM3
get_TM4
get_TM5
get_RMS
get_V
get_LOG
get_WL
get_AAC
get_DAMV
get_DASDV
get_ZC
get_MYOP
get_WAMP
get_SSC
get_AR
get_CC
get_MNF
get_MDF
get_PKF
get_MNP
get_TTP
get_SM1
get_SM2
get_SM3
get_VCF
"""


def get_skewse(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    return skew(frame)


def get_IEMG(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    IEMG = np.sum(np.abs(frame))
    return IEMG


def get_MAV(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    MAV = np.sum(np.abs(frame)) / N

    return MAV


def get_MAV1(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    w = np.zeros(N)
    w[int(0.25 * N) : int(0.75 * N)] = 1
    w[: int(0.25 * N)] = 0.5
    w[int(0.75 * N) :] = 0.5
    MAV1 = np.sum(w * np.abs(frame)) / N

    return MAV1


def get_MAV2(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    w = np.zeros(N)
    w[int(0.25 * N) : int(0.75 * N)] = 1
    w[: int(0.25 * N)] = 4 * np.arange(int(0.25 * N)) / N
    w[int(0.75 * N) :] = 4 * (np.arange(N - int(0.75 * N)) - N) / N
    MAV2 = np.sum(w * np.abs(frame)) / N

    return MAV2


def get_SSI(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    SSI = np.sum(frame * frame)
    return SSI


def get_VAR(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    VAR = np.sum(np.power(frame, 2)) / (N - 1)

    return VAR


def get_TM3(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    TM3 = np.abs(np.sum(np.power(frame, 3))) / N

    return TM3


def get_TM4(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    TM4 = np.sum(np.power(frame, 4)) / N

    return TM4


def get_TM5(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    TM5 = np.abs(np.sum(np.power(frame, 5))) / N

    return TM5


def get_RMS(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    RMS = np.sqrt(np.sum(np.power(frame, 2)) / N)

    return RMS


def get_V(frame, v=2):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    V = np.power(np.sum(np.power(frame, v)) / N, 1 / v)

    return V


def get_LOG(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    frame = np.where(frame < 1e-5, 1e-5, frame)  # Try except で書く
    N = frame.shape[0]
    LOG = np.exp(np.sum(np.log(np.abs(frame))) / N)
    return LOG


def get_WL(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    WL = np.sum(np.abs(np.diff(frame)))
    return WL


def get_AAC(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    AAC = np.sum(np.abs(np.diff(frame))) / N
    return AAC


def get_DAMV(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    DAMV = np.sum(np.diff(frame)) / (N - 1)
    return DAMV


def get_DASDV(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    DASDV = np.sqrt(np.sum(np.power(np.diff(frame), 2)) / (N - 1))

    return DASDV


def get_ZC(frame, threshold=0):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    zero_crossing = np.nonzero(np.diff(frame >= threshold))[0]
    return zero_crossing.size


def get_MYOP(frame, threshold=0.05):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    N = frame.shape[0]
    MYOP = np.sum(frame >= threshold) / N
    return MYOP


def get_WAMP(frame, threshold=0.005):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    diff = np.abs(np.diff(frame))
    WAMP = np.sum(diff >= threshold)
    return WAMP


def get_SSC(frame, threshold=0.00016):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    diff1 = np.diff(frame)
    diff2 = np.roll(diff1, -1)
    SSC = np.nonzero(diff1[:-1] * diff2[:-1] >= threshold)[0]

    return SSC.size


def get_AR(frame, P=4):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    a = lfilter(np.ones(P), np.array([1, *np.zeros(P - 1)]), frame)
    print(a.shape)
    return a


def get_CC(frame, P=4):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    a = lfilter(np.ones(P), np.array([1, *np.zeros(P - 1)]), frame)
    c = np.zeros(P)
    c[0] = -a[0]

    for p in range(1, P):
        for l in range(1, p):
            c[p] -= (1 - l / p) * a[l] * c[p - l]
        c[p] -= a[p]
    return c[P - 1]


def get_MNF(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2
    MNF = np.sum(f * spectrum) / np.sum(spectrum)

    return MNF


def get_MDF(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2
    total_power = np.sum(spectrum)
    half_power = total_power / 2
    running_power = 0
    MDF = 0
    for j in range(N):
        running_power += spectrum[j]
        if running_power >= half_power:
            MDF = f[j]
            break

    return MDF


def get_PKF(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2
    PKF = f[np.argmax(spectrum)]

    return PKF


def get_MNP(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    spectrum = np.abs(np.fft.fft(frame)) ** 2
    MNP = np.mean(spectrum)

    return MNP


def get_TTP(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    spectrum = np.abs(np.fft.fft(frame)) ** 2
    TTP = np.sum(spectrum)

    return TTP


def get_SM1(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2

    SM1 = np.sum(spectrum * f)

    return SM1


def get_SM2(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2

    SM2 = np.sum(spectrum * f**2)

    return SM2


def get_SM3(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2

    SM3 = np.sum(spectrum * f**3)

    return SM3


def get_VCF(frame, fs=1024):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    N = frame.shape[0]
    f = np.fft.fftfreq(N, d=1 / fs)
    spectrum = np.abs(np.fft.fft(frame)) ** 2

    SM0 = get_TTP(frame)
    SM1 = get_SM1(frame, fs)
    SM2 = get_SM2(frame, fs)

    f_c = SM1 / SM0
    VCF = SM2 / SM0 - f_c**2

    return VCF
