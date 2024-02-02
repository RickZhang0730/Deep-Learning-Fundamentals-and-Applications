import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import loadmat
PATH = "data.mat"
data = sio.loadmat(r"C:\Users\rick\Desktop\HW1\112_HW1\data.mat")

x=data["x"]
y=data["y"]

plt.plot(x, y)
plt.show()

print(y.shape)
print(x.shape)

