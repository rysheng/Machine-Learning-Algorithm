import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#
# def get_data():
#     data = pd.read_csv('data/the_trang_logistic_regression.csv')
#     X = data[['TheTrang']].values
#     y = data[['Loai']].values
#     return X, y
#
# def draw_Xy(X, y):
#     plt.scatter(X, y, c=y)
#
# def draw_show(title='', xlabel='', ylabel=''):
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()
#
# X, y = get_data()
#
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
# X_scaler = min_max_scaler.fit_transform(X)
# scaler_saver = [min_max_scaler.min_, min_max_scaler.scale_]
# scaler_saver
#
# ones = np.ones((X_scaler.shape[0], 1))
# X_bar = np.concatenate([ones, X_scaler], axis=1)
#
# w = np.random.random(size=(X_bar.shape[1], 1))
# learning_rate = 1.0
# epochs = 2000
#
# for ep in range(epochs):
#     index_random = np.random.permutation(X_bar.shape[0])
#     for i in index_random:
#         xi = X_bar[i].reshape(1, -1)
#         yi_hat = 1 / (1 + np.exp(-xi.dot(w)))
#         w -= learning_rate * (yi_hat - y[i]) * xi.T
#
# # predict
# x_new = 175
# mm_scaler = MinMaxScaler(feature_range=(-1, 1))
# mm_scaler.min_ = scaler_saver[0]
# mm_scaler.scale_ = scaler_saver[1]
# x_new_scaler = mm_scaler.transform(np.array([[x_new]]))
# x_new_bar = np.array([[1, x_new_scaler[0][0]]])
# y_predict = 1 / (1 + np.exp(-x_new_bar.dot(w)))
#
# print(y_predict)
# if y_predict > 0.8:
#     print('Loai 1')
# elif y_predict < 0.2:
#     print('Loai 0')
# else:
#     print('Không xác định')

data = np.array([[1, 2], [-0.5, 6], [0, 10], [-1, 18]])
scaler = MinMaxScaler(feature_range=(-1, 1))
print(scaler.fit_transform(data))
print(data.min(axis=0))
print(scaler.min_)
print(scaler.scale_)
print(scaler.data_max_)
#[ 0.   -1.25]
# [1.    0.125]