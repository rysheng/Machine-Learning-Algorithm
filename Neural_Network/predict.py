from Neural_Network.models import  scaler
from sklearn.preprocessing import MinMaxScaler
from Neural_Network.neural_network import soft_max
import numpy as np

x_new = np.array([[175, 50]])
min_max_scaler = MinMaxScaler()
min_max_scaler.min_ = scaler.scaler[0]
min_max_scaler.scale_ = scaler.scaler[1]
x_new_scaler = min_max_scaler.transform(x_new)

W1 = np.load('models/w1.npy')
b1 = np.load('models/b1.npy')
W2 = np.load('models/w2.npy')
b2 = np.load('models/b2.npy')

z1 = x_new_scaler.dot(W1) + b1
a1 = np.maximum(z1, 0)
z2 = a1.dot(W2)+b2
y_predict = soft_max(z2)
print(y_predict)