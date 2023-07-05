import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def calc_scores(y_pred_label, y_pred_prob, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    my_auc = auc(fpr, tpr)
    # fnr = 1 - tpr
    # #eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    my_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(my_eer)

    cm = confusion_matrix(y_true, y_pred_label)
    tn, fp, fn, tp = cm.flatten()

    my_accuracy_score = accuracy_score(y_true, y_pred_label)
    my_error_rate = 1 - my_accuracy_score
    my_recall_score = recall_score(y_true, y_pred_label, zero_division=0)
    my_precision_score = precision_score(y_true, y_pred_label, zero_division=0)
    my_f1_score = f1_score(y_true, y_pred_label, zero_division=0)
    spec = tn / (tn + fp)

    result = {
        "accuracy": my_accuracy_score,
        "error_rate": my_error_rate,
        "recall_score": my_recall_score,
        "precision_score": my_precision_score,
        "f1_score": my_f1_score,
        "eer": my_eer,
        "eer_thresh": thresh,
        "auc": my_auc,
        "specificity": spec,
    }

    return result
