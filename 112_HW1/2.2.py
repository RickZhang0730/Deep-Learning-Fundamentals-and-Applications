import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the data from the .mat file
data = loadmat("data.mat")
x = data["x"]
y = data["y"]

# Initialize lists to store coefficients for 200 iterations
theta_samples = []

# Number of random samples to draw
num_samples = 200

# Perform 200 iterations
for _ in range(num_samples):
    # Randomly select 30 data samples
    random_indices = np.random.choice(len(x), size=30, replace=False)
    x_sampled = x[random_indices]
    y_sampled = y[random_indices]
    
    # Fit a quartic curve (polynomial of degree 4) using numpy's polyfit
    theta = np.polyfit(x_sampled.flatten(), y_sampled.flatten(), 4)
    
    # Extract the coefficients
    theta_samples.append(theta)

# Plot the 200 quartic curves
plt.figure(figsize=(10, 6))
for i in range(num_samples):
    y_pred_sample = np.polyval(theta_samples[i], x.flatten())
    plt.plot(x, y_pred_sample, color='red', alpha=0.2)  # Use alpha to make lines semi-transparent

# Original data points
plt.scatter(x, y, color='#3D7878')

plt.legend()
plt.title(f'200 Random Quartic Curves')
plt.show()