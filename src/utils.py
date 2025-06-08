import numpy as np
from data_loader import load_svm_file
from sklearn.model_selection import train_test_split
from CDPER_algorithm import CDPER_L2SVM

X, y = load_svm_file('../data/paper_data/real-sim.binary')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grand_truth_model = CDPER_L2SVM(C=1, max_iter=1000, random_state=42, exact_hessian=False, tol=10)
grand_truth_model.fit(X_train, y_train)
f_w_star = grand_truth_model.objective_values[-1][1]
np.save('grand_truth_value.npy', f_w_star)
print(f_w_star)
