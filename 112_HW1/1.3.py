import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.rcParams['figure.figsize'] = (20.0, 20.0)

# input data
data = loadmat(r"C:\Users\rick\Desktop\HW1\112_HW1\data.mat")
x = data["x"]
y = data["y"]

# Building the model
X = np.column_stack((x, x**2))  # Create a feature matrix with x and x^2
X = np.column_stack((np.ones_like(x), X))  # Add a column of ones for the intercept

# Calculate the coefficients using the normal equation
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Extract the coefficients
theta_0 = theta[0]
theta_1 = theta[1]
theta_2 = theta[2]

print("Theta_0 is: ", theta_0)
print("Theta_1 is: ", theta_1)
print("Theta_2 is: ", theta_2)

# Make predictions
y_pred = theta_0 + x * theta_1 + x**2 * theta_2

# Calculate Mean Squared Error
mse = ((y - y_pred) ** 2).mean()
print("Error in least square parabola :", mse)

# Showing model data
plt.scatter(x, y)
plt.scatter(x, y_pred, color='red')
plt.show()