import numpy as np
import matplotlib.pyplot as plt
from  sklearn.svm import LinearSVC


class CoordinateDescentSVM:
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, sigma=0.01, beta=0.5):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.beta = beta
        self.w = None

    def _d_prime_i_0(self, X, y, i):
        margins = 1 - y * (X @ self.w)
        active = margins > 0
        bj = margins[active]
        xji = X[active, i]
        yj = y[active]
        gradient_sum = np.sum(yj * xji * bj)
        return self.w[i] - 2 * self.C * gradient_sum

    def _d_double_prime_i_0(self, X, y, i):
        margins = 1 - y * (X @ self.w)
        active = margins > 0
        xji = X[active, i]
        gradient_sum = np.sum(xji * xji)
        return 1 + 2 * self.C * gradient_sum

    def _newton_direction(self, X, y, i):
        numerator = self._d_prime_i_0(X, y, i)
        denominator = self._d_double_prime_i_0(X, y, i)
        return -numerator / denominator if denominator != 0 else 0.0

    def _d_i_z(self, X, y, i, z):
        ei = np.zeros(X.shape[1])
        ei[i] = 1
        w_new = self.w + z * ei
        margins = 1 - y * (X @ w_new)
        active = margins > 0
        bj = margins[active]
        loss_term = np.sum(bj ** 2)
        return 0.5 * np.dot(w_new, w_new) + self.C * loss_term

    def _compute_lambda(self, X, y, i, d):
        D0 = self._d_i_z(X, y, i, 0)
        z = d
        D_z = self._d_i_z(X, y, i, z)
        threshold = D_z - D0

        k = 1
        while threshold > -self.sigma * (z ** 2):
            z = d * (self.beta ** k)
            D_z = self._d_i_z(X, y, i, z)
            threshold = D_z - D0
            k += 1
        return self.beta ** (k - 1)

    def _coordinate_update(self, X, y, i):
        d = self._newton_direction(X, y, i)
        lam = self._compute_lambda(X, y, i, d)
        self.w[i] += lam * d

    def fit(self, X, y):
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w = np.zeros(X_ext.shape[1])

        for iteration in range(self.max_iter):
            w_old = self.w.copy()
            print(iteration)
            for i in range(X_ext.shape[1]):
                self._coordinate_update(X_ext, y, i)
                print(i)

            if np.linalg.norm(self.w - w_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        return self

    def predict(self, X):
        return np.sign(X @ self.w)
