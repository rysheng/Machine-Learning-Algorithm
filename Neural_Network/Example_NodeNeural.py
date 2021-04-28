import numpy as np

data = np.array([[0,0,0,1],
                 [1,1,1,1],
                 [1,0,1,1],
                 [0,1,1,0]])



X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

w = np.random.random((3, 1)) * 2 - 1
learning_rate = 0.1

for i in range(200):
    y_hat = 1 / (1 + np.exp(-np.dot(X, w)))
    loss = y_hat - y
    w -= learning_rate * np.dot(X.T, loss * y_hat * (1 - y_hat))


# predict
x_new = np.array([1,0,0])
y_new = 1 / (1+ np.exp(-x_new.dot(w)))
print(y_new)