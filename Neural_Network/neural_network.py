import numpy as np
from Neural_Network import reprocess
from sklearn.preprocessing import MinMaxScaler

X, y = reprocess.get_data()

min_max_scaler = MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
reprocess.draw_Xy(X_scaler, y)
reprocess.draw_show()

def soft_max(z):
    ep = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ep / np.sum(ep, axis=1, keepdims=True)

def Loss_function(y_hat, y):
    return -np.mean(np.log(y_hat[range(y_hat.shape[0]), y]))

def neural_fit(X, y, w1, b1, w2, b2, learning_rate, epochs):
    loss_hist = []
    for ep in range(epochs):
        z1 = X.dot(w1) + b1
        a1 = np.maximum(z1, 0)  # relu
        z2 = a1.dot(w2) + b2
        y_hat = soft_max(z2) # sortmax
        if (ep % 50 == 0):
            loss = Loss_function(y_hat, y)
            print(f'iter: {ep} | loss: {loss}')
            loss_hist.append(loss)
        # back propagation
        y_hat[range(y_hat.shape[0]), y] -= 1
        e2 = y_hat / X.shape[0]
        dw2 = np.dot(a1.T, e2)
        db2 = np.sum(e2, axis=0)
        e1 = np.dot(e2, W2.T)
        e1[z1 <= 0] = 0
        dw1 = np.dot(X.T, e1)
        db1 = np.sum(e1, axis=0)

        # gradient descent
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

    return (w1, b1), (w2, b2), loss_hist

learning_rate = 0.01
epochs = 1000

d0 = 2    # features
d1 = 20  # nodes number of hidden layer
d2 = 6    # classifies number

W1 = np.random.random((d0, d1))
b1 = np.zeros(d1)

W2 = np.random.random((d1, d2))
b2 = np.zeros(d2)

(W1, b1), (W2, b2), loss_hist = neural_fit(X=X_scaler, y=y, w1=W1, b1=b1, w2=W2, b2=b2, learning_rate=0.1, epochs=1000)
reprocess.draw_line(list(range(0, epochs, 50)), loss_hist)
reprocess.draw_show(title='Neural Network', xlabel='epochs', ylabel='loss')

np.save('models/w1', W1)
np.save('models/b1', b1)
np.save('models/w2', W2)
np.save('models/b2', b2)




