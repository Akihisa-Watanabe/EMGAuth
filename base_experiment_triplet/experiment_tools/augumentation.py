import numpy as np
from audiomentations import AddGaussianNoise, Compose, Shift


def augumentation(X_train, y_train):
    X_train = X_train.astype(np.float32)
    if len(X_train.shape) == 2:
        X_train = X_train[np.newaxis, :]

    transforms_1 = Compose([Shift(min_fraction=-0.25, max_fraction=0.25, p=1)])
    transforms_2 = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1)])
    augmented_data_1 = transforms_1(samples=X_train, sample_rate=1000)
    augmented_data_2 = transforms_2(samples=X_train, sample_rate=1000)
    augmented_data = np.concatenate([augmented_data_1, augmented_data_2], axis=1)
    augmented_y = np.concatenate([y_train, y_train], axis=0)

    X_train = np.concatenate([X_train, augmented_data], axis=1)
    y_train = np.concatenate([y_train, augmented_y], axis=0)
    if len(X_train.shape) == 3:
        X_train = np.squeeze(X_train, axis=0)
    return X_train, y_train
