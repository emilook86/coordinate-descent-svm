import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Tkagg')

f_w_star = np.load('grand_truth_value.npy')
model1_objective_values = np.load('model1.objective_values.npy')
model1_gradient_values = np.load('model1.gradient_values.npy')
model2_objective_values = np.load('model2.objective_values.npy')
model2_gradient_values = np.load('model2.gradient_values.npy')

print(model1_gradient_values)
def plotting(x1_values, y1_values, x2_values, y2_values, title, x_axis_name, y_axis_name, algorithm1, algorithm2):

    # Create the plot
    plt.figure(figsize=(8, 6))  # Optional: set figure size
    plt.plot(x1_values, y1_values, 'b-', label=f'{algorithm1}')  # 'b-' means blue solid line
    plt.plot(x2_values, y2_values, 'r--', label=f'{algorithm2}')

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
plotting(x1_values,
         y1_values,
         x2_values,
         y2_values,
         'Gradient values with respect to time',
         'Time [s]',
         'Gradient ||w||',
         'CDPER',
         "CDPER without exact Hessian")

x1_values = [point[0] for point in model1_objective_values]
y1_values = [(point[1] - f_w_star) / (np.abs(f_w_star)) for point in model1_objective_values]
x2_values = [point[0] for point in model2_objective_values]
y2_values = [(point[1] - f_w_star) / (np.abs(f_w_star)) for point in model2_objective_values]
plotting(x1_values,
         y1_values,
         x2_values,
         y2_values,
         'Relative value to the minimum with respect to the time',
         'Time [s]',
         'Relative function value difference',
         'CDPER',
         "CDPER without exact Hessian")
