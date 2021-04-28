import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Sortmax_Regression import reprocess


X, y = reprocess.get_data()

# reprocess.draw_Xy(X, y)
# reprocess.draw_show('Chiều Cao', 'Cân Nặng', 'SoftMax_Regression')

min_max_scaler = MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
# print(min_max_scaler.min_, min_max_scaler.scale_)

ones = np.ones(shape=(X.shape[0], 1))
X_bar = np.concatenate((ones, X_scaler), axis=1)

class_num = 6
w = np.random.random(size=(X_bar.shape[1], class_num))

def softmax(z):
    ez = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ez / np.sum(ez, axis=1, keepdims=True)

def Loss_gradient(X, y, w):
    s = softmax(X.dot(w))
    return X.T.dot(s-y) / X.shape[0]


def softmax_loss(X, y, w):
    s = softmax(X.dot(w))
    return np.mean(np.sum(-np.log(s)*y, axis=1))

eye_class = np.eye(class_num)
y_one_hot = eye_class[y.reshape(-1)]

epochs = 1000
learning_rate = 0.1
batch_size = 20
losses = []

for _ in range(epochs):
    X_bar_random = np.random.permutation(X_bar.shape[0])
    for i in range(int(np.ceil(X_bar_random.shape[0] / batch_size))):
        batch_i = X_bar_random[i*batch_size: min((i+1)*batch_size, X_bar.shape[0])]
        xi_loss = X_bar[batch_i]
        yi_loss = y_one_hot[batch_i]
        w -= learning_rate*Loss_gradient(X=xi_loss, y=yi_loss, w=w)

    loss = softmax_loss(X=X_bar, y=y_one_hot, w=w)
    losses.append(loss)

# reprocess.draw_line(range(epochs), losses)
# reprocess.draw_show('epochs', 'loss', 'SoftMax_Regression')
np.save('models/the_trang_models.npy', w)


