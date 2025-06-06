import numpy as np
from scipy.sparse import issparse, csc_matrix
import time
from data_loader import load_svm_file
from sklearn.model_selection import train_test_split


class CDPER_L2SVM:
    def __init__(self, C=1.0, sigma=0.01, beta=0.5, max_iter=1000, tol=1e-4, random_state=42, exact_hessian=True):
        self.C = C
        self.sigma = sigma
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.exact_hessian = exact_hessian
        self.w = None
        self.z = None
        self.H = None
        self.lambdas = {}
        np.random.seed(random_state)

    def _precompute_H(self, X):
        """
            Precompute diagonal elements of Hessian matrix:
            H_i = 1 + 2C * sum_j x_ji^2
            Used for second derivatives during coordinate updates.
        """
        self.H = 1 + 2 * self.C * (X.power(2).sum(axis=0)).A1

    def _get_active_mask(self, y):
        """
            Identify samples violating margin (i.e., 1 - y*z > 0).
            Only these contribute to gradient updates.
        """
        return (1 - y * self.z) > 0

    def _d_prime_i(self, X, y, i):
        """
            Compute gradient (first derivative) w.r.t. w[i]
        """
        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        indices = X.indices[col_start:col_end]
        data = X.data[col_start:col_end]

        active_mask = self._get_active_mask(y)[indices]
        if not np.any(active_mask):
            return self.w[i]

        y_active = y[indices[active_mask]]
        z_active = self.z[indices[active_mask]]
        data_active = data[active_mask]

        margins = 1 - y_active * z_active
        return self.w[i] - 2 * self.C * np.sum(data_active * y_active * margins)

    def _d_double_prime_i(self, X, y, i, exact=False):
        """
        Return the second derivative (Hessian diagonal) for feature i.

        exact:  If True, compute using only active set (violating examples)
                If False (default), return precomputed H[i] as upper bound
        """
        if not exact:
            return self.H[i]

        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        indices = X.indices[col_start:col_end]
        data = X.data[col_start:col_end]

        active_mask = self._get_active_mask(y)[indices]
        if not np.any(active_mask):
            return 1.0

        data_active = data[active_mask]
        return 1.0 + 2 * self.C * np.sum(data_active ** 2)

    def _newton_direction(self, X, y, i):
        """
            Compute Newton direction for coordinate i:
            direction = -gradient / hessian
        """
        d_prime = self._d_prime_i(X, y, i)
        d_double_prime = self._d_double_prime_i(X, y, i, exact=self.exact_hessian)
        return -d_prime / d_double_prime if d_double_prime != 0 else 0.0

    def _line_search(self, X, y, i, d):
        """
            Perform backtracking line search to ensure sufficient objective decrease:
            D(w + λd) - D(w) <= -σ * (λd)^2
        """
        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        indices = X.indices[col_start:col_end]
        data = X.data[col_start:col_end]

        D0 = self._objective(y=y)

        lam = 1.0
        while True:
            delta = lam * d

            z_new = self.z.copy()
            z_new[indices] += delta * data

            D_new = self._objective(w=self.w, z=z_new, y=y)

            if (D_new - D0) <= -self.sigma * (delta ** 2):
                return lam
            lam *= self.beta
            if lam < 1e-10:
                return lam

    def _objective(self, w=None, z=None, y=None):
        """
            Compute the primal objective:
            0.5 * ||w||^2 + C * sum((1 - y*z)_+^2)
        """
        if w is None:
            w = self.w
        if z is None:
            z = self.z
        if y is None:
            raise ValueError("y must be provided")

        margins = 1 - y * z
        loss = np.sum(margins[margins > 0] ** 2)
        return 0.5 * np.dot(w, w) + self.C * loss

    def _compute_constant_lambda(self, X, y, i):
        """
            Precompute λ̄ upper bound from the line search theorem:
            λ̄ = d'' / (0.5 * H_i + σ)
        """
        dii = self._d_double_prime_i(X, y, i, exact=self.exact_hessian)

        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        data = X.data[col_start:col_end]
        Xi_squared_sum = np.sum(data ** 2)

        Hi = 1 + 2 * self.C * Xi_squared_sum
        return dii / (0.5 * Hi + self.sigma), dii

    def fit(self, X, y):
        if not issparse(X) or not isinstance(X, csc_matrix):
            X = csc_matrix(X)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.z = X @ self.w
        self._precompute_H(X)

        for k in range(self.max_iter):
            perm = np.random.permutation(n_features)
            w_old = self.w.copy()

            start_time = time.time()
            inner_iter = 0

            for i in perm:
                inner_iter += 1

                if i not in self.lambdas:
                    self.lambdas[i], _ = self._compute_constant_lambda(X, y, i)

                d = self._newton_direction(X, y, i)
                if abs(d) < 1e-10:
                    continue

                lam = self._line_search(X, y, i, d)
                delta = lam * d
                self.w[i] += delta

                col_start = X.indptr[i]
                col_end = X.indptr[i + 1]
                indices = X.indices[col_start:col_end]
                data = X.data[col_start:col_end]
                self.z[indices] += delta * data

                if inner_iter % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Inner iteration {inner_iter}, error = {1 - self.score(X, y):.6f}. Time elapsed: {elapsed:.2f} s.")

            delta_w = np.linalg.norm(self.w - w_old)
            if k % 2 == 0 or k == self.max_iter - 1:
                print(f"Iter {k}, Δw = {delta_w:.2f}, error = {1 - self.score(X, y):.6f}")
            if delta_w < self.tol:
                break
        return self

    def predict(self, X):
        if issparse(X):
            return np.sign(X @ self.w)
        return np.sign(np.dot(X, self.w))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


X, y = load_svm_file('../data/paper_data/news20.binary')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model = CDPER_L2SVM(C=1.0, max_iter=1000, random_state=42)
#model.fit(X_train, y_train)
model2 = CDPER_L2SVM(C=1.0, max_iter=1000, random_state=42, exact_hessian=False)
model2.fit(X_train, y_train)
