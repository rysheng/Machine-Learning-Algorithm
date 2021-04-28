from Linear_Regression import process
import numpy as np
import time

def gradient(X, Y, w):
    y_hat = X.dot(w)
    return np.dot(X.T, y_hat - Y) / X.shape[0]


X, Y = process.read_data('data/the_trang_linear_regression.csv')
x_min, x_max = np.min(X), np.max(X)

# chuẩn hóa dữ liệu
X_scale = (X.reshape(-1, 1) - x_min) / (x_max - x_min) # reshape(-1, 1): điều chỉnh lại len(X)-1 dòng và 1 cột
y_min, y_max = np.min(Y), np.max(Y)
Y_scale = (Y.reshape(-1, 1) - y_min) / (y_max - y_min)

ones = np.ones((X.shape[0], 1))
X_bar = np.concatenate((ones, X_scale), axis=1)


w = np.random.random(size=(X_bar.shape[1], 1))

epochs = 5000
learning_rate = 0.1
for epoch in range(epochs):
    w -= learning_rate * gradient(X=X_bar, Y=Y_scale, w=w)

process.draw_x(X_scale, Y_scale)
# y = w0 + w1 * x
x_draw = np.array([0.0, 1.0])
y_draw = w[0][0] + x_draw * w[1][0]
process.draw_line(x_draw, y_draw)
process.draw_show()


np.save('model/model_the_trang_linear_rg.npy', w)
print(w)