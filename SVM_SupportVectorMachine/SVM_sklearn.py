from sklearn.svm import SVC
import numpy as np
from SVM_SupportVectorMachine.reprocess import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

X, y = get_data()

model = SVC()
model.fit(X, y)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(model, X, y):
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, cmap=plt.cm.tab20, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    return out


plot_contours(model, X, y)