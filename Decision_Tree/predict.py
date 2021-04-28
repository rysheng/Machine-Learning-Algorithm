import numpy as np
import pickle
from Decision_Tree.models.encode_saver import encode_saver

f = open('models/play_decision_models', 'rb')
model = pickle.load(f)
f.close()


x_new = ['sunny', 75, 75, True]

x_new[0] = encode_saver['OUTLOOK'].index(x_new[0])
x_new[3] = encode_saver['WINDY'].index(x_new[3])

x_new_np = np.array([x_new])
y_pred = model.predict(x_new_np)
y_pred_lb = encode_saver['Play'][y_pred[0]]
print(y_pred_lb)
