import numpy as np
from scipy.spatial.distance import cdist
from Logistict_Regression import reproccess
from sklearn.preprocessing import MinMaxScaler

X, y = reproccess.get_data()

# x_min, x_max = np.min(X), np.max(X)
# X_scale = 2*(X.reshape(-1, 1) - x_min) / (x_max - x_min)-1
# print(x_min, x_max)
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X_scale = min_max_scaler.fit_transform(X)
print(min_max_scaler.min_, min_max_scaler.scale_)
# reproccess.draw_Xy(X_scale, y)
# reproccess.draw_show()


ones = np.ones((X_scale.shape[0], 1))
X_bar = np.concatenate((ones, X_scale), axis=1)
y = y.reshape(-1, 1)
w = np.random.random((X_bar.shape[1], 1))

learning_rate = 0.01
epochs = 1000
batch_size = 20
loss = []
for ep in range(epochs):
    index_random = np.random.permutation(X_bar.shape[0])
    total_loss = 0
    for i in range(int(X.shape[0]/batch_size)):
        batch_i = index_random[i*batch_size: min((i+1)*batch_size, X.shape[0])]
        xi = X_bar[batch_i]
        yi = y[batch_i]
        yi_hat = 1/(1+np.exp(-xi.dot(w)))
        w -= learning_rate*xi.T.dot(yi_hat-yi)
        total_loss += np.sum((yi-yi_hat)**2)
    loss.append(np.sqrt(total_loss) / X.shape[0])
np.save('models/the_models_logistic_regression.npy', w)
print(w)
# reproccess.draw_line(range(epochs), loss)
# reproccess.draw_show()