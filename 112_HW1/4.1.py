import keras
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

test = loadmat(r"C:\Users\rick\Desktop\HW1\112_HW1\test.mat")
train = loadmat(r"C:\Users\rick\Desktop\HW1\112_HW1\train.mat")

def setimage(dataset):
    X_zip = zip(dataset['x1'], dataset['x2'])
    X = np.array(list(X_zip))
    X = X.reshape((X.shape[0], X.shape[1]))
    Y = dataset['y']
    Y = Y.reshape(Y.shape[0]) 
    return X, Y

train_x, train_y = setimage(train)
test_x, test_y = setimage(test)

# input image dimensions 28x28
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def sampling(x, y):
    ynew = np.array([])
    for i in range(10):
        label_locate = np.where(y == i)
        label_locate = np.array(label_locate[0])
        random.shuffle(label_locate)
        ynew = np.hstack([ynew, np.array([i]*500)])
        if i == 0:
            xnew = x[label_locate[:500]]
        else:
            xnew = np.vstack([xnew, x[label_locate[:500]]])
    return np.array(xnew), np.array(ynew)

my_x_data, my_y_data = sampling(x_train, y_train)
indexarr = np.array(range(5000))
random.shuffle(indexarr)
rnd_my_y_data = my_y_data[indexarr]
rnd_my_x_data = my_x_data[indexarr]
amount = 50
lines = 5
columns = 10
number = np.zeros(amount)
for i in range(amount):
    number[i] = rnd_my_y_data[i]

    fig = plt.figure()
for i in range(amount):
    ax = fig.add_subplot(lines, columns, 1 + i)
    plt.imshow(rnd_my_x_data[i, :, :], cmap='binary')
    plt.sca(ax)
    ax.set_xticks([], [])
    ax.set_yticks([], [])

[plt.close(f) for f in plt.get_fignums() if f != 50]
plt.show()