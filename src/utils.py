import numpy as np
from data_loader import load_svm_file
from sklearn.model_selection import train_test_split
from CDPER_algorithm import CDPER_L2SVM
from scipy.sparse import csc_matrix

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = csc_matrix(X_train)
X_test = csc_matrix(X_test)

grand_truth_model = CDPER_L2SVM(C=1, max_iter=1000, random_state=42, max_time=5*seconds)
grand_truth_model.fit(X_train, y_train)
f_w_star = grand_truth_model.objective_values[-1][1]
np.save(f'grand_truth_{data}_value.npy', f_w_star)
print(f_w_star)
