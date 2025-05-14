import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def plotting(model, X, y):

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Parametry granicy decyzyjnej: w1*x1 + w2*x2 + b = 0 → x2 = -(w1*x1 + b)/w2
    w = model.w[:-1]
    b = model.w[-1]

    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    y_vals = -(w[0] * x_vals + b)

    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('CoordinateDescentSVM – Granica decyzyjna')
    plt.legend()
    plt.grid(True)
    plt.show()
