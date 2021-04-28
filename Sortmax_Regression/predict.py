from Sortmax_Regression.models.scaler import scaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Sortmax_Regression.softmax import softmax
w = np.load('models/the_trang_models.npy')


x_new = np.array([[180, 60]])


min_max_scaler = MinMaxScaler()
min_max_scaler.min_, min_max_scaler.scale_ = scaler[0], scaler[1]
x_new_scaler = min_max_scaler.transform(x_new)

ones = np.ones((x_new_scaler.shape[0], 1))
x_new_bar = np.concatenate([ones, x_new_scaler], axis=1)
y_predict = softmax(x_new_bar.dot(w))
print(y_predict)