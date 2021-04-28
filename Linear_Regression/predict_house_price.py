import numpy as np
from Linear_Regression.model.scaler import  scaler_house_X, scaler_house_Y

#     ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'condition', 'sqft_above', 'sqft_basement']
# x_predict = [4.0, 2.5, 2920.0, 4000.0, 1.5, 5.0, 1910.0, 1010.0]
x_predict = [3.0, 1.5, 1340.0, 7912.0, 1.5, 3.0, 1340.0, 0.0]
x_predict_scale = (x_predict + scaler_house_X[0]) / (scaler_house_X[1] - scaler_house_X[0])
x_predict_scale = np.insert(x_predict_scale, 0, 1)

w = np.load('model/model_house_price_linear_rg.npy')

y_predict_scale = x_predict_scale.dot(w)

predict_price_house = y_predict_scale * (scaler_house_Y[1] - scaler_house_Y[0]) + scaler_house_Y[0]

print(f'{float(predict_price_house)} $')
