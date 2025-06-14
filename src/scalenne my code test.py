import numpy as np
import time
from scipy.sparse import issparse, csc_matrix
import datetime
import os

def load_svm_file(file_path, zero_based=True):
    labels = []
    rows = []
    cols = []
    data = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            labels.append(float(parts[0]))

            for feat in parts[1:]:
                idx, val = feat.split(':')
                idx = int(idx) - (0 if zero_based else 1)
                rows.append(i)
                cols.append(idx)
                data.append(float(val))

    # Jawna konwersja do CSC
    from scipy.sparse import coo_matrix
    X = coo_matrix((data, (rows, cols))).tocsc()
    y = np.array(labels)

    return X, y 

class SparseCoordinateDescentSVM_2:
    def __init__(self, C=1.0, max_iter=10000, tol=1e-8, sigma=0.01, beta=0.5, verbose=True):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose
        self.w = None
        self.z = None
        self.H = None
        self.lambdas = {}
        self.times = []
        self.relative_diffs = []
        self.obj_values =[]
        self.time_start = time.time()



    def _precompute_H(self, X):
        """
            Precompute diagonal elements of Hessian matrix:
            H_i = 1 + 2C * sum_j x_ji^2
            Used for second derivatives during coordinate updates.
        """
        return  1 + 2 * self.C * (X.power(2).sum(axis=0)).A1
    def _d_prime_i_0(self, X, y, i):
        margins = 1 - y * self.z
        active = margins > 0

        # Indeksy kolumny i w formacie CSC
        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        row_indices = X.indices[col_start:col_end]
        data = X.data[col_start:col_end]

        # Maska aktywnych przykładów w tych wierszach
        mask = active[row_indices]

        if not np.any(mask):
            return self.w[i]  # brak aktywnych przykładów, gradient to tylko w[i]

        filtered_y = y[row_indices][mask]
        filtered_margins = margins[row_indices][mask]
        filtered_data = data[mask]

        gradient_sum = np.sum(filtered_data * filtered_y * filtered_margins)
        return self.w[i] - 2 * self.C * gradient_sum


    def _d_double_prime_i_0(self, X, y, i):
        # Wyciągamy niezerowe elementy kolumny i z csc_matrix X
        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        indices = X.indices[col_start:col_end]  # wiersze niezerowych elementów kolumny i
        data = X.data[col_start:col_end]       # wartości tych elementów

        # Obliczamy marginesy
        margins = 1 - y * self.z
        # Tworzymy maskę aktywnych przykładów (tych, które łamią warunek margin > 0)
        active_mask = margins[indices] > 0

        if not np.any(active_mask):
            return 1.0  # tylko regularizacja, brak strat

        data_active = data[active_mask]

        return 1.0 + 2 * self.C * np.sum(data_active ** 2)


    def _newton_direction(self, X, y, i,  denominator = None):

        numerator = self._d_prime_i_0(X, y, i)

        if (denominator == None):
            denominator = self._d_double_prime_i_0(X, y, i)
        return -numerator / denominator if denominator != 0 else 0.0

    def _d_i_z(self, X, y, i, z):
        x_col = X[:, i]
        indices = x_col.indices
        delta = z * x_col.data
        z_new_part = self.z[indices] + delta
    
        margins_part = 1 - y[indices] * z_new_part
        active = margins_part > 0
        loss_term = np.sum(margins_part[active] ** 2)
    
        w_norm_sq = np.dot(self.w, self.w) + 2 * z * self.w[i] + z**2
        return 0.5 * w_norm_sq + self.C * loss_term


    def _compute_threshhold_lambda(self, X, y, i):
  
        dii = self._d_double_prime_i_0(X, y, i)

        return dii / (0.5 * self.H[i]+ self.sigma), dii



    def _compute_lambda(self, X, y, i, d):
        lambda_bar = self.lambdas[i]
        if abs(d) < 1e-12:
            return 0.0
        if 1.0 <= lambda_bar:
            return 1.0

        # Otherwise, perform line search
        D0 = self._d_i_z(X, y, i, 0)
        k = 0
        while True:
            lam = self.beta ** k
            z = lam * d
            Dz = self._d_i_z(X, y, i, z)
            if Dz - D0 <= -self.sigma * (z ** 2):
                return lam
            k += 1
            if k > 20:  # Prevent infinite loop
                return lam

    def _coordinate_update(self, X, y, i, z, dii=None):
        d = self._newton_direction(X, y, i, dii)
        if abs(d) < 1e-12:
            return

        lam = 1.0
        if self.lambdas[i] < 1.0:  # Only do line search if needed
            lam = self._compute_lambda(X, y, i, d)

        delta = lam * d
        self.w[i] += delta

        # Efficient update of self.z for sparse column
        col_start = X.indptr[i]
        col_end = X.indptr[i + 1]
        indices = X.indices[col_start:col_end]
        data = X.data[col_start:col_end]

        self.z[indices] += delta * data



    def fit(self, X, y):
        print(f"start  time = {datetime.datetime.now()}")
        if not issparse(X) or not isinstance(X, csc_matrix):
            raise ValueError("X must be a CSC (Compressed Sparse Column) matrix")
        #start with initial w0
        
        self.w = np.zeros(X.shape[1],dtype=np.float64)
        self.H = self._precompute_H(X)
        self.z = X.dot(self.w)
        
        for iteration in range(self.max_iter):
            i = np.random.randint(0, X.shape[1])
            if i not in self.lambdas:
                self.lambdas[i], dii = self._compute_threshhold_lambda(X, y, i)
            else:
                dii = None
            self._coordinate_update(X, y, i, dii)
            
            if iteration % 1000 == 0 or iteration == self.max_iter - 1:
                print(f"Iter {iteration}, error = {1-self.score(X,y):.3e}, time = {datetime.datetime.now()}")

                elapsed_time = time.time() - self.time_start
                self.times.append(elapsed_time)
    

        print(f"finish  time = {datetime.datetime.now()}")
        return self


    def predict(self, X):
        return np.sign(X @ self.w)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
        
    def _objective(self, X, y):
        margins = 1 - y * self.z
        loss = np.sum((margins[margins > 0]) ** 2)
        return 0.5 * np.dot(self.w, self.w) + self.C * loss

model = SparseCoordinateDescentSVM_2(C=1.0, max_iter=11355192
                                     )
X, y = load_svm_file('C:/Users/danci/source/repos/coordinate-descent-svm-master/data/paper_data/news20.binary')

model.fit(X,y)