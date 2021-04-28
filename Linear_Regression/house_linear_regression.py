from Linear_Regression import process
import numpy as np

def gradient(X, Y, w):
    y_hat = X.dot(w)
    return np.dot(X.T, y_hat - Y) / X.shape[0]

X, Y = process.read_data('data/house_price.csv')

ones = np.ones((X.shape[0], 1))
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)

X_scale = (X - x_min) / (x_max - x_min)
y_min, y_max = np.min(Y), np.max(Y)
Y_scale = (Y - y_min) / (y_max - y_min)
X_bar = np.concatenate((ones, X_scale), axis=1)

A = X_bar.T.dot(X_bar)
B = X_bar.T.dot(Y_scale)

w = np.random.random(size=(X_bar.shape[1], 1))

epochs = 5000
learning_rate = 0.1
for epoch in range(epochs):
    w -= learning_rate * gradient(X=X_bar, Y=Y_scale, w=w)

print(w)
np.save('model/model_house_price_linear_rg.npy', w)