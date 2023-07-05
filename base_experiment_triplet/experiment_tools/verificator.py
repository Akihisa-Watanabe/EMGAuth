import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity


class Verify(BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.enroll_data = None

    def fit(self, X_enroll, y=None):
        self.enroll_data = X_enroll
        return self

    def predict(self, X_test):
        if self.enroll_data is None:
            raise ValueError(
                "Enrollment data is not set. Call 'fit' method with enrollment data first."
            )

        if isinstance(X_test, torch.Tensor):
            if X_test.requires_grad:
                X_test_np = X_test.detach().numpy()
                X_enroll_np = self.enroll_data.detach().numpy()
            else:
                X_test_np = X_test.numpy()
                X_enroll_np = self.enroll_data.numpy()

        else:
            X_test_np = X_test

        similarity = cosine_similarity(X_enroll_np, X_test_np)
        avg_similarity = similarity.mean(axis=0)

        return (avg_similarity > self.threshold).astype(int)

    def predict_similarity(self, X_test):
        if self.enroll_data is None:
            raise ValueError(
                "Enrollment data is not set. Call 'fit' method with enrollment data first."
            )

        if isinstance(X_test, torch.Tensor):
            if X_test.requires_grad:
                X_test_np = X_test.detach().numpy()
                X_enroll_np = self.enroll_data.detach().numpy()
            else:
                X_test_np = X_test.numpy()
                X_enroll_np = self.enroll_data.numpy()

        else:
            X_test_np = X_test

        similarity = cosine_similarity(X_enroll_np, X_test_np)
        avg_similarity = similarity.mean(axis=0)
        return avg_similarity

    def score(self, X_test, y_test, optimize=False):
        if optimize:
            self.threshold = self.optimize_threshold(X_test, y_test)

        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    def objective(self, threshold, X_test, y_test):
        self.threshold = threshold
        accuracy = self.score(X_test, y_test)
        return -accuracy

    def optimize_threshold(self, X_test, y_test):
        result = minimize_scalar(
            self.objective, bounds=(-1, 1), args=(X_test, y_test), method="bounded"
        )
        best_threshold = result.x
        return best_threshold
