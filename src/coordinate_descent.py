import numpy as np
from scipy.sparse import issparse


class SparseCoordinateDescentSVM:
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, sigma=0.01, beta=0.5):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.beta = beta
        self.w = None

    def _d_prime_i_0(self, X, y, i):
        margins = 1 - y * (X @ self.w)  # Sparse-friendly
        active = margins > 0
        bj_active = margins[active]
        xji = X[active, i]  # Sparse column slice (CSC efficient)
        y_active = y[active]

        # Compute gradient without converting to dense
        if issparse(xji):
            # Sparse-safe: y_j * X_ji * b_j
            gradient_sum = (xji.multiply(y_active * bj_active)).sum()
        else:
            gradient_sum = np.sum(y_active * xji * bj_active)

        return self.w[i] - 2 * self.C * gradient_sum

    def _d_double_prime_i_0(self, X, y, i):
        margins = 1 - y * (X @ self.w)  # Sparse-friendly
        active = margins > 0
        xji = X[active, i]  # Sparse column slice (CSC efficient)

        # Compute Hessian term without converting to dense
        if issparse(xji):
            gradient_sum = (xji.power(2)).sum()  # Sum(X_ji²)
        else:
            gradient_sum = np.sum(xji ** 2)

        return 1 + 2 * self.C * gradient_sum

    def _newton_direction(self, X, y, i):
        numerator = self._d_prime_i_0(X, y, i)
        denominator = self._d_double_prime_i_0(X, y, i)
        return -numerator / denominator if denominator != 0 else 0.0

    def _d_i_z(self, X, y, i, z):
        ei = np.zeros(X.shape[1])
        ei[i] = 1
        w_new = self.w + z * ei
        margins = 1 - y * (X @ w_new)  # Sparse-friendly
        active = margins > 0
        bj_active = margins[active]
        loss_term = np.sum(bj_active ** 2)
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
        self.w = np.zeros(X.shape[1])  # Dense (sparse w would complicate updates)

        for iteration in range(self.max_iter):
            print(iteration)
            w_old = self.w.copy()
            for i in range(X.shape[1]):
                self._coordinate_update(X, y, i)
                print(i)

            if np.linalg.norm(self.w - w_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        return self

    def predict(self, X):
        return np.sign(X @ self.w)  # Sparse-friendly

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
