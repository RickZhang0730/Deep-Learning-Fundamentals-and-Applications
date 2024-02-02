import keras
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    Y = Y.reshape(Y.shape[0])  # 將(n,1)轉(n,)
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
    newy = np.array([])
    for i in range(10):
        label_locate = np.where(y == i)
        label_locate = np.array(label_locate[0])
        random.shuffle(label_locate)
#         print('the random index of label %d in dataset is :\n' %i,label_locate[:100],'\n')
        newy = np.hstack([newy, np.array([i]*500)])
#         print('the label %d of train data is :\n' %i ,x[label_locate[:100]])
        if i == 0:
            newx = x[label_locate[:500]]
        else:
            newx = np.vstack([newx, x[label_locate[:500]]])
    return np.array(newx), np.array(newy)


samplex, sampley = sampling(x_train, y_train)

indexarr = np.array(range(5000))
random.shuffle(indexarr)

# 由於經過抽樣後的資料會照數字大小排序，故將資料排序打亂
randy = sampley[indexarr]
randx = samplex[indexarr]

randx = randx.reshape(5000, 28*28).astype('float32')

n_components = 2
pca = PCA(n_components=n_components).fit(randx)

eigenvalues = pca.components_.reshape(n_components, 28, 28)
# eigenvalues = pca.components_

# 畫圖
plt.figure(figsize=(8, 8))
for i in list(range(2)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(eigenvalues[i].reshape(28, 28), cmap='jet')
    title_text = 'Eigenvalue ' + str(i + 1)
    plt.title(title_text, size=6.5)
    plt.xticks(())
    plt.yticks(())
plt.legend()
plt.show()
