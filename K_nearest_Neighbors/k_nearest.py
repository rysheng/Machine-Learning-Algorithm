import numpy as np
from K_nearest_Neighbors import process
from scipy.spatial.distance import cdist

print(np.random.seed(7))
data = process.get_data('data/the_trang.csv')
data_label = process.get_data('data/the_trang_labels.csv')

X = data[:, [0, 1]]
Y = data[:, 2]
k = 7

infor = [175, 40]
new_infor = np.array([infor])
D = cdist(X, new_infor)
k_nearest = Y[np.argsort(D, axis=0)[:k]]
classes, count = np.unique(k_nearest, return_counts=True)
guess = classes[np.argmax(count)]

human_fit = data_label[guess][data_label.shape[1]-1]
print("Than hinh cua ban la: " + human_fit)