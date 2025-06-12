import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Tkagg')

dataset = 2     # pick a number 1-4

if dataset == 1:
    data = 'news20'

if dataset == 2:
    data = 'real-sim'

if dataset == 3:
    data = 'rcv1_test'

if dataset == 4:
    data = 'synthetic1'

f_w_star = np.load(f'grand_truth_{data}_value.npy')
model1_objective_values = np.load(f'model1_{data}_objective_values.npy')
model1_gradient_values = np.load(f'model1_{data}_gradient_values.npy')
model1_accuracy_values = np.load(f'model1_{data}_accuracy_values.npy')
model2_objective_values = np.load(f'model2_{data}_objective_values.npy')
model2_gradient_values = np.load(f'model2_{data}_gradient_values.npy')
model2_accuracy_values = np.load(f'model2_{data}_accuracy_values.npy')
model3_objective_values = np.load(f'model3_{data}_objective_values.npy')
model3_gradient_values = np.load(f'model3_{data}_gradient_values.npy')
model3_accuracy_values = np.load(f'model3_{data}_accuracy_values.npy')

def plotting(x1_values, y1_values, x2_values, y2_values, x3_values, y3_values,
             title, x_axis_name, y_axis_name, algorithm1, algorithm2, algorithm3, log=False):

    # Create the plot
    plt.figure(figsize=(8, 6))  # Optional: set figure size
    plt.plot(x1_values, y1_values, 'b-', label=f'{algorithm1}')  # 'b-' means blue solid line
    plt.plot(x2_values, y2_values, 'r--', label=f'{algorithm2}')
    plt.plot(x3_values, y3_values, 'g:', label=f'{algorithm3}')

    if log:
        plt.yscale('log')

    # Add labels and title
    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')
    plt.title(f'{title}')
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: add grid
    plt.savefig(f'{'_'.join(title.split())}.png')
    plt.show()


x1_values = [point[0] for point in model1_gradient_values]
y1_values = [point[1] for point in model1_gradient_values]
x2_values = [point[0] for point in model2_gradient_values]
y2_values = [point[1] for point in model2_gradient_values]
x3_values = [point[0] for point in model3_gradient_values]
y3_values = [point[1] for point in model3_gradient_values]
plotting(x1_values,
         y1_values,
         x2_values,
         y2_values,
         x3_values,
         y3_values,
         f'Gradient values with respect to time, data {data}',
         'Time [s]',
         'Gradient ||w||',
         'CDPER',
         "CDPER without exact Hessian",
         'Acceleration',
         log=True)


x1_values = [point[0] for point in model1_objective_values]
y1_values = [(point[1] - f_w_star) / (np.abs(f_w_star)) for point in model1_objective_values]
x2_values = [point[0] for point in model2_objective_values]
y2_values = [(point[1] - f_w_star) / (np.abs(f_w_star)) for point in model2_objective_values]
x3_values = [point[0] for point in model3_objective_values]
y3_values = [(point[1] - f_w_star) / (np.abs(f_w_star)) for point in model3_objective_values]
plotting(x1_values,
         y1_values,
         x2_values,
         y2_values,
         x3_values,
         y3_values,
         f'Relative value to the minimum with respect to the time, data {data}',
         'Time [s]',
         'Relative function value difference',
         'CDPER',
         "CDPER without exact Hessian",
         'Acceleration',
         log=True)


x1_values = [point[0] for point in model1_accuracy_values]
y1_values = [point[1] for point in model1_accuracy_values]
x2_values = [point[0] for point in model2_accuracy_values]
y2_values = [point[1] for point in model2_accuracy_values]
x3_values = [point[0] for point in model3_accuracy_values]
y3_values = [point[1] for point in model3_accuracy_values]
plotting(x1_values,
         y1_values,
         x2_values,
         y2_values,
         x3_values,
         y3_values,
         f'Accuracy on test data with respect to time, data {data}',
         'Time [s]',
         'Accuracy',
         'CDPER',
         "CDPER without exact Hessian",
         'Acceleration')