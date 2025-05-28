from benchmarks import *
from coordinate_descent import *
from data_loader import *
from utils import *
from sklearn.svm import LinearSVC
import os
from pathlib import Path

model_our = SparseCoordinateDescentSVM()

current_dir = Path(__file__).parent  # Goes to folder src
project_root = current_dir.parent    # Goes to project root
data_path = project_root / "data" / "paper_data" / "real-sim.binary"

X_train, X_test, y_train, y_test = load_dataset(data_path)
X_train = X_train[:200, 1:3]
y_train = y_train[:200]
X_test = X_test[:200, 1:3]
y_test = y_test[:200]

print("X_train type:", type(X_train))
print("X_train shape:", X_train.shape)
print("X_train ndim:", X_train.ndim)
print(X_train)
print(y_train)

model_our2 = SparseCoordinateDescentSVM(tol=1e-6)
model_our2.fit(X_train, y_train)
a = model_our2.score(X_test, y_test)
print(a)

#plotting(model_our2, X_train, y_train)

current_dir = Path(__file__).parent  # Goes to folder src
project_root = current_dir.parent    # Goes to project root
data_path = project_root / "data" / "synthetic_data" / "synthetic1.binary"

X_train, X_test, y_train, y_test = load_dataset(data_path)

model_our3 = SparseCoordinateDescentSVM(tol=1e-6)
model_our3.fit(X_train, y_train)
b = model_our3.score(X_test, y_test)
print(b)
