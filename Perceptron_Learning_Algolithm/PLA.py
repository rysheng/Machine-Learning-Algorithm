import numpy as np
from Perceptron_Learning_Algolithm import process
from sklearn.preprocessing import MinMaxScaler
import time

X, y = process.get_data('data/the_trang_binary_classification.csv')


min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X = min_max_scaler.fit_transform(X)

ones = np.ones((X.shape[0], 1))
X_bar = np.concatenate([ones, X], axis=1)

w = 2*np.random.random((X_bar.shape[1], 1))-1
learning_rate = 1.0
y_hat = np.dot(X_bar, w)
loss_index = ((y_hat > 0) ^ (y > 0)).reshape(-1)
X_loss = X_bar[loss_index]

while True:
    y_hat = np.dot(X_bar, w)
    loss_index = ((y_hat > 0) ^ (y > 0)).reshape(-1)
    X_loss = X_bar[loss_index]
    y_loss = y[loss_index]
    if X_loss.shape[0] > 0:
        x_check = X_loss[0]
        y_check = y_loss[0]
        w += learning_rate * x_check.reshape(-1, 1) * y_check
        x_draw = np.array([-1,  1])
        y_draw = -(w[0][0] + w[1][0] * x_draw) / w[2][0]

        process.draw_line(x_draw, y_draw)
        process.draw_points(X=X, y=y)
        process.draw_show()
        time.sleep(1)
    else:
        break

loss_index = ((y_hat > 0) ^ (y > 0)).reshape(-1)
print(loss_index)
