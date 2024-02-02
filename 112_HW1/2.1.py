import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the data from the .mat file
data = loadmat("data.mat")
x = data["x"]
y = data["y"]

# Initialize lists to store coefficients for 200 iterations
theta0_samples = []
theta1_samples = []

# Number of random samples to draw
num_samples = 200

# Perform 200 iterations
for _ in range(num_samples):
    # Randomly select 30 data samples
    random_indices = np.random.choice(len(x), size=30, replace=False)
    x_sampled = x[random_indices]
    y_sampled = y[random_indices]
    
    # Build the feature matrix
    X_sampled = np.column_stack((np.ones_like(x_sampled), x_sampled))
    
    # Calculate the coefficients using the normal equation
    theta = np.linalg.inv(X_sampled.T.dot(X_sampled)).dot(X_sampled.T).dot(y_sampled)
    
    # Extract the coefficients
    theta0_samples.append(theta[0])
    theta1_samples.append(theta[1])

# Plot the 200 regression lines
plt.figure(figsize=(10, 6))
for i in range(num_samples):
    y_pred_sample = theta0_samples[i] + x * theta1_samples[i]
    plt.plot(x, y_pred_sample, color='red', alpha=0.2)  # Use alpha to make lines semi-transparent

# Original data points
plt.scatter(x, y, color='#3D7878')

plt.legend()
plt.title(f'200 Random Linear Regression Lines')
plt.show()