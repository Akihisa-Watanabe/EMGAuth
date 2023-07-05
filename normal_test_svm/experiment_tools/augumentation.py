import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audiomentations import AddGaussianNoise, Compose, Shift, TimeStretch


def augumentation(X_train, y_train):
    transforms_0 = Compose(
        [
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        ]
    )
    transforms_1 = Compose([Shift(min_fraction=-0.25, max_fraction=0.25, p=1.0)])
    transforms_2 = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0)])

    augmented_data_0 = transforms_0(samples=X_train.values, sample_rate=1000)
    augmented_data_1 = transforms_1(samples=X_train.values, sample_rate=1000)
    augmented_data_2 = transforms_2(samples=X_train.values, sample_rate=1000)

    augmented_data = np.concatenate([augmented_data_0, augmented_data_1, augmented_data_2], axis=0)
    augmented_y = np.concatenate([y_train, y_train, y_train], axis=0)

    # graph show
    # plt.plot(X_train[y_train==1].T, alpha=0.7, color='blue')
    # plt.plot(augmented_data[augmented_y==1].T, alpha=0.1, color='red')
    # plt.show()

    # concat
    X_train = np.concatenate(
        [X_train, augmented_data], axis=0
    )  # X_train_aug_shift,X_train_aug_stretch
    y_train = np.concatenate([y_train, augmented_y], axis=0)  # y_train, y_train

    return X_train, y_train
