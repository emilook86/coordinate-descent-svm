import numpy as np
import time
from scipy.sparse import issparse, csc_matrix
import datetime
from line_profiler import LineProfiler

class Coordinate_boosted():
    def __init__(self, C=1.0, max_iter=10000, tol=1e-8, sigma=0.01, beta=0.5, verbose=True):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.gamma = 0
        self.beta = beta
        self.verbose = verbose
        self.w = None
        self.z = None
        self.k = 0

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

    def _d_double_prime_i_0(self, X, y, i, exact = True):


        if not exact:
            return self.H[i]
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



    def fit(self, X, y_labels):
        if not issparse(X) or not isinstance(X, csc_matrix):
            raise ValueError("X must be a CSC (Compressed Sparse Column) matrix")

        self.n = X.shape[1]
        self.x = np.zeros(self.n, dtype=np.float64)  # x0 docelowo szukany wektor

        self.v = self.x.copy()
        self.Xx_cashed = X.dot(self.x)
        self.Xv_cashed = X @ self.v 
        self.y_cashed = np.zeros(self.n, dtype=np.float64)
        self.H = self._precompute_H(X)
        gamma_k = 0.0
        gamma_prev = gamma_k
        i = np.random.randint(0, self.n,self.max_iter)
        iter = 0
        for k in i:
            # -- 1. Wyznacz gamma_k rozwiązując równanie kwadratowe -- check
            #gamma_prev = gamma_k
            B = (self.sigma * gamma_prev**2 - 1) / self.n
            C = -gamma_prev**2

            discriminant = B**2 - 4 * C
            gamma_k = (-B + np.sqrt(discriminant)) / 2
            
             # -- 2. Oblicz alfa, beta -- check
            alpha_k = (self.n - gamma_k * self.sigma) / (gamma_k * (self.n**2 - self.sigma))
            beta_k = 1 - (gamma_k * self.sigma) / self.n

            # -- 4. Wybierz losowy indeks --

            col_start = X.indptr[k]
            col_end = X.indptr[k + 1]
            row_indices = X.indices[col_start:col_end]  # wiersze niezerowych elementów kolumny i
            data = X.data[col_start:col_end] 


            # -- 3. y_k = α_k * v + (1 - α_k) * w -- check

            #self.y_cashed  = alpha_k * self.v + (1 - alpha_k) * self.x
            y_k_value = alpha_k * self.v[k] + (1 - alpha_k) * self.x[k]

                        

            # -- 5. Licz gradient po y_k (czyli po ∇f(y)[i]) --
            margins = 1 - y_labels * self.Xx_cashed
            active = margins > 0




            z_y_rows = alpha_k * self.Xv_cashed[row_indices] + (1 - alpha_k) * self.Xx_cashed[row_indices]
            margins_rows = 1 - y_labels[row_indices] * z_y_rows
            mask = margins_rows > 0



            if not np.any(mask):
                grad_i = y_k_value
            else:
                filtered_data = data[mask]                             # X_{j, i_k}
                filtered_y = y_labels[row_indices][mask]               # y_j
                filtered_margins = margins_rows[mask]                  # (1 - y_j * z_j)
                grad_sum = np.sum(filtered_data * filtered_y * filtered_margins)
                grad_i = y_k_value - 2 * self.C * grad_sum


            # -- 6. Lipschitz dla i-tego kierunku (aproksymacja Hessianu) --
            dii = self._d_double_prime_i_0(X, y_labels, k, exact = False)

            # -- 7. Krok po i-tej współrzędnej (update x[i]) --
            delta_i = -grad_i / dii
            self.x[k] += delta_i
            self.Xx_cashed[row_indices] += delta_i * data

            # -- 8. Update v (momentum) --
            old_vi = self.v[k]
            self.v[k] = beta_k * self.v[k] + (1 - beta_k) * y_k_value - (gamma_k / dii) * grad_i
            self.Xv_cashed[row_indices] += (self.v[k] - old_vi) * data

            # -- 9. Zaktualizuj gamma_prev --
            gamma_prev = gamma_k
            
            # -- 10. Monitoring co 100 iteracji --
            if iter % 10000 == 0:
                acc = self.score(X, y_labels)
                print(f"[{iter:5d}] acc = {acc:.4f} , time = {datetime.datetime.now()}")
            iter +=1

    def predict(self, X):
        return np.sign(X @ self.x)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
            
    def _objective(self, X, y):
        margins = 1 - y * (X@self.x)
        loss = np.sum((margins[margins > 0]) ** 2)
        return 0.5 * np.dot(self.x, self.x) + self.C * loss
    

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

X, y = load_svm_file('../data/paper_data/news20.binary')

# Model
model = Coordinate_boosted(C=1, tol=0, max_iter=X.shape[1])

# Opakowanie w funkcję do profilowania
def profiled_fit():
    model.fit(X, y)

# Profilowanie
lp = LineProfiler()
lp.add_function(model.fit)  # dodajemy metodę do profilera
lp_wrapper = lp(profiled_fit)  # opakowujemy funkcję globalną
lp_wrapper()  # wywołanie

# Raport
lp.print_stats()

# Można też jeszcze raz bez profilowania:
#print("\n=== Wersja bez profilowania ===")
#print(datetime.datetime.now())
model.fit(X, y)