import numpy as np
from Linear_Regression.model.scaler import scaler

x_new = 170
w = np.load('model/model_the_trang_linear_rg.npy')

x_new_norm = (x_new - scaler[0][0]) / (scaler[0][1] - scaler[0][0])
# y = w0 + w1 * x
y_pred = w[0][0] + x_new_norm * w[1][0]
y_pred_show = y_pred * (scaler[1][1] - scaler[1][0]) + scaler[1][0]
print(y_pred_show)