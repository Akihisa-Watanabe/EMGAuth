import pandas as pd

from .my_features import *

"""
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
get_VCF"""


def feature_extractor(X, myop_thre, wamp_thre, ssc_thre, cc_P):
    df = pd.DataFrame(X)

    SKEWSE = df.apply(func=get_skewse, axis=1)
    IEMG = df.apply(func=get_IEMG, axis=1)
    MAV = df.apply(func=get_MAV, axis=1)
    MAV1 = df.apply(func=get_MAV1, axis=1)
    MAV2 = df.apply(func=get_MAV2, axis=1)
    SSI = df.apply(func=get_SSI, axis=1)
    VAR = df.apply(func=get_VAR, axis=1)
    TM3 = df.apply(func=get_TM3, axis=1)
    TM4 = df.apply(func=get_TM4, axis=1)
    TM5 = df.apply(func=get_TM5, axis=1)
    RMS = df.apply(func=get_RMS, axis=1)
    V = df.apply(func=get_V, axis=1)
    LOG = df.apply(func=get_LOG, axis=1)
    WL = df.apply(func=get_WL, axis=1)
    AAC = df.apply(func=get_AAC, axis=1)
    DAMV = df.apply(func=get_DAMV, axis=1)
    DASDV = df.apply(func=get_DASDV, axis=1)
    ZC = df.apply(func=get_ZC, axis=1)
    MYOP = df.apply(func=get_MYOP, axis=1, threshold=myop_thre)
    WAMP = df.apply(func=get_WAMP, axis=1, threshold=wamp_thre)
    SSC = df.apply(func=get_SSC, axis=1, threshold=ssc_thre)
    # AR = df.apply(func=get_AR, axis=1)
    CC = df.apply(func=get_CC, axis=1, P=cc_P)
    MNF = df.apply(func=get_MNF, axis=1)
    MDF = df.apply(func=get_MDF, axis=1)
    PKF = df.apply(func=get_PKF, axis=1)
    MNP = df.apply(func=get_MNP, axis=1)
    TTP = df.apply(func=get_TTP, axis=1)
    SM1 = df.apply(func=get_SM1, axis=1)
    SM2 = df.apply(func=get_SM2, axis=1)
    SM3 = df.apply(func=get_SM3, axis=1)
    VCF = df.apply(func=get_VCF, axis=1)

    feature_names = [
        "skewe",
        "iemg",
        "mav",
        "mav1",
        "mav2",
        "ssi",
        "var",
        "tm3",
        "tm4",
        "tm5",
        "rms",
        "v",
        "log",
        "wl",
        "aac",
        "damv",
        "dasdv",
        "zc",
        "myop",
        "wamp",
        "ssc",
        "cc",
        "mnf",
        "mdf",
        "pkf",
        "mnp",
        "ttp",
        "sm1",
        "sm2",
        "sm3",
        "vcf",
    ]

    df_emg_fatures = pd.DataFrame(columns=feature_names)

    df_emg_fatures["skewe"] = SKEWSE
    df_emg_fatures["iemg"] = IEMG
    df_emg_fatures["mav"] = MAV
    df_emg_fatures["mav1"] = MAV1
    df_emg_fatures["mav2"] = MAV2
    df_emg_fatures["ssi"] = SSI
    df_emg_fatures["var"] = VAR
    df_emg_fatures["tm3"] = TM3
    df_emg_fatures["tm4"] = TM4
    df_emg_fatures["tm5"] = TM5
    df_emg_fatures["rms"] = RMS
    df_emg_fatures["v"] = V
    df_emg_fatures["log"] = LOG
    df_emg_fatures["wl"] = WL
    df_emg_fatures["aac"] = AAC
    df_emg_fatures["damv"] = DAMV
    df_emg_fatures["dasdv"] = DASDV
    df_emg_fatures["zc"] = ZC
    df_emg_fatures["myop"] = MYOP
    df_emg_fatures["wamp"] = WAMP
    df_emg_fatures["ssc"] = SSC
    # df_emg_fatures['ar'] = AR
    df_emg_fatures["cc"] = CC
    df_emg_fatures["mnf"] = MNF
    df_emg_fatures["mdf"] = MDF
    df_emg_fatures["pkf"] = PKF
    df_emg_fatures["mnp"] = MNP
    df_emg_fatures["ttp"] = TTP
    df_emg_fatures["sm1"] = SM1
    df_emg_fatures["sm2"] = SM2
    df_emg_fatures["sm3"] = SM3
    df_emg_fatures["vcf"] = VCF
    return df_emg_fatures
