import numpy as np
from scipy.sparse import issparse, csc_matrix
import time
from data_loader import load_svm_file
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


class CDPER_L2SVM:
    def __init__(self, C=1.0, sigma=0.01, beta=0.5, max_iter=1000, tol=1e-4, random_state=42, exact_hessian=True, max_time=70):
        self.C = C
        self.sigma = sigma
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.exact_hessian = exact_hessian
        self.max_time = max_time
        self.w = None
        self.z = None
        self.H = None
        self.lambdas = {}

        self.objective_values = []
        self.gradient_values = []
        self.gradient_norm_values = []
        self.accuracies = []

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
        d_prime_i = self.w[i] - 2 * self.C * np.sum(data_active * y_active * margins)
        self.gradient_values[i] = d_prime_i
        return d_prime_i

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

    def _line_search(self, X, y, i, d, lambda_bar):
        """
            Perform backtracking line search to ensure sufficient objective decrease:
            D(λd) - D(0) <= -σ * (λd)^2
        """
        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        indices = X.indices[col_start:col_end]
        data = X.data[col_start:col_end]

        D0 = self._objective(y=y)

        lam = 1.0
        while True:
            if lam <= lambda_bar:
                return lam  # skip evaluating D_i(λd), inequality is guaranteed

            # Otherwise evaluate the objective difference
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

    def _compute_gradient(self, X, y):
        """Compute full gradient ∇f(w) = w + 2C * X^T(y * (1 - y*Xw)_+"""
        active_mask = self._get_active_mask(y)
        margins = (1 - y * self.z) * active_mask
        return self.w + 2 * self.C * X.T.dot(y * margins)

    def fit(self, X, y, Xtest=None, ytest=None):
        if not issparse(X) or not isinstance(X, csc_matrix):
            X = csc_matrix(X)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.z = X @ self.w
        self.gradient_values = np.ones(n_features)
        self._precompute_H(X)

        start_even_before = time.time()
        for k in range(self.max_iter):
            perm = np.random.permutation(n_features)
            w_old = self.w.copy()

            inner_iter = 0

            for i in perm:
                inner_iter += 1

                if i not in self.lambdas:
                    self.lambdas[i], _ = self._compute_constant_lambda(X, y, i)

                d = self._newton_direction(X, y, i)
                if abs(d) < 1e-10:
                    continue

                lam = self._line_search(X, y, i, d, lambda_bar=self.lambdas[i])
                delta = lam * d
                self.w[i] += delta

                col_start = X.indptr[i]
                col_end = X.indptr[i + 1]
                indices = X.indices[col_start:col_end]
                data = X.data[col_start:col_end]
                self.z[indices] += delta * data

                elapsed = time.time() - start_even_before
                if inner_iter % 10000 == 9999:
                    f_w = self._objective(y=y)
                    self.objective_values.append((elapsed, f_w))
                    print(f"Inner iteration {inner_iter}, error = {1 - self.score(X, y):.6f}. Time elapsed: {elapsed:.2f} s.")

                    final_grad_norm = np.linalg.norm(self.gradient_values)
                    self.gradient_norm_values.append((elapsed, final_grad_norm))

                    if Xtest is not None and ytest is not None:
                        accuracy = self.score(Xtest, ytest)
                        self.accuracies.append((elapsed, accuracy))

                if elapsed > self.max_time:
                    return self

            elapsed = time.time() - start_even_before
            final_grad_norm = np.linalg.norm(self.gradient_values)
            self.gradient_norm_values.append((elapsed, final_grad_norm))

            if final_grad_norm < self.tol:  # Now using gradient norm for convergence
                print(f"Converged at iter {k+1}, grad_norm = {final_grad_norm:.4f}")
                break

            if Xtest is not None and ytest is not None:
                accuracy = self.score(Xtest, ytest)
                print(f"Accuracy on test set at iter {k+1}: {accuracy:.4f}")
                self.accuracies.append((elapsed, accuracy))


            print(f"Exited outer iteration loop number {k+1}, grad_norm = {final_grad_norm:.4f}, time: {elapsed:.2f} s.")
            print(f"And moreover, objective value function: {self.objective_values[-1][1]}")
        return self

    def predict(self, X):
        if issparse(X):
            return np.sign(X @ self.w)
        return np.sign(np.dot(X, self.w))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {'C': self.C, 'sigma': self.sigma, 'beta': self.beta,
                'max_iter': self.max_iter, 'tol': self.tol,
                'random_state': self.random_state,
                'exact_hessian': self.exact_hessian}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


# CHANGE HERE
dataset = 1     # pick a number 1-4

if dataset == 1:
    X, y = load_svm_file('../data/paper_data/news20.binary')
    data = 'news20'
    seconds = 400

if dataset == 2:
    X, y = load_svm_file('../data/paper_data/real-sim.binary')
    data = 'real-sim'
    seconds = 200

if dataset == 3:
    X, y = load_svm_file('../data/paper_data/rcv1_test.binary')
    data = 'rcv1_test'
    seconds = 500

if dataset == 4:
    X, y = load_svm_file('../data/synthetic_data/synthetic1.binary')
    data = 'synthetic1'
    seconds = 200

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = csc_matrix(X_train)
    X_test = csc_matrix(X_test)

    #grand_truth_model = CDPER_L2SVM(C=1, max_iter=1000, random_state=42, exact_hessian=False, tol=100)
    #grand_truth_model.fit(X_train, y_train)
    #f_w_star = grand_truth_model.objective_values[-1][1]
    #np.save('grand_truth_value.npy', f_w_star)


    model1 = CDPER_L2SVM(C=1.0, max_iter=1000, random_state=42, max_time=seconds)
    model1.fit(X_train, y_train, X_test, y_test)
    score_model1 = model1.score(X_test, y_test)
    print(f"Accuracy: {score_model1}")
    print(model1.objective_values)
    print(model1.gradient_values)
    np.save(f'model1_{data}_objective_values.npy', model1.objective_values)
    np.save(f'model1_{data}_gradient_values.npy', model1.gradient_norm_values)
    np.save(f'model1_{data}_accuracy_values.npy', model1.accuracies)

    model2 = CDPER_L2SVM(C=1.0, max_iter=1000, random_state=42, exact_hessian=False, max_time=seconds)
    model2.fit(X_train, y_train, X_test, y_test)
    score_model2 = model2.score(X_test, y_test)
    print(f"Accuracy: {score_model2}")
    print(model2.objective_values)
    print(model2.gradient_norm_values)
    np.save(f'model2_{data}_objective_values.npy', model2.objective_values)
    np.save(f'model2_{data}_gradient_values.npy', model2.gradient_norm_values)
    np.save(f'model2_{data}_accuracy_values.npy', model2.accuracies)


    """
    param_grid = {'C': [0.1, 0.5, 1, 5, 10]}
    model = CDPER_L2SVM(random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    print("Test accuracy: {:.4f}".format(best_model.score(X_test, y_test)))
    """