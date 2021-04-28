import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Logistict_Regression.models.scaler import scaler
w = np.load('models/the_models_logistic_regression.npy')

x_new = 175

# x_new = np.array([[x_new]])
# x_new_scaler = 2*(x_new - scaler[0])/(scaler[1]-scaler[0])-1
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
minmax_scaler.min_ = scaler[0]
minmax_scaler.scale_ = scaler[1]
x_new_scaler = minmax_scaler.transform(np.array([[x_new]]))

x_new_bar = np.array([[1, x_new_scaler[0][0]]])
y_predict = 1 / (1+np.exp(-x_new_bar.dot(w)))
print(y_predict)
if y_predict > 0.8:
    print('Loai 1')
elif y_predict < 0.2:
    print('Loai 0')
else:
    print('Không xác định')
