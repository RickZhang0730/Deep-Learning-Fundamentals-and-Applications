from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


test = loadmat(r"test.mat")
train = loadmat(r"train.mat")


def preprocess(dataset):
    zip1 = zip(dataset['x1'], dataset['x2'])
    X = np.array(list(zip1))
    X = X.reshape((X.shape[0], X.shape[1]))
    Y = dataset['y']
    Y = Y.reshape(Y.shape[0])  # 將(n,1)轉(n,)
    return X, Y


train_x, train_y = preprocess(train)
test_x, test_y = preprocess(test)

m, n = train_x.shape
theta = np.zeros(n)
theta = theta.reshape(n, 1)

inte = np.ones(m)
inte = inte.reshape(m, 1)
train_x1 = np.append(inte, train_x, axis=1)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_x[:, 0], train_x[:, 1], train_y)
plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def log_likelihood(theta, x, y):
    j = np.sum(-y*np.log(sigmoid(np.dot(x, theta))) - (1-y)
               * (np.log(1-sigmoid(np.dot(x, theta)))))/m
    return j


l = log_likelihood(theta, train_x, train_y)


def gradientDescent(x, y, theta, alpha, iterations):
    for i in range(iterations):
        gradient = np.dot(x.transpose(), (sigmoid(
            np.dot(x, theta))-y.reshape(m, 1)))/m
        theta -= (alpha/m) * gradient
    return theta


theta = gradientDescent(train_x, train_y, theta, 0.01, 10000)
print(theta)

print('MSE is ', mse(test_y.reshape(30, 1), test_x.dot(theta)))
print('L1 loss is ', np.sum(np.abs(test_y.reshape(30, 1)-test_x.dot(theta))))

model = LogisticRegression(fit_intercept=False)
model.fit(train_x, train_y)
print(model.intercept_, model.coef_)

mse(test_y, model.predict(test_x))

model.score(test_x, test_y)
