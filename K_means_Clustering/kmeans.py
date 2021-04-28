from K_means_Clustering import process
import numpy as np
from scipy.spatial.distance import cdist
import time

np.random.seed(7)
X = process.read_data(path="data/the_trang_kmeans.csv")
k = 3

center = X[np.random.choice(X.shape[0], k, replace=False)]
process.draw_x(x=X[:, 0], y=X[:, 1])
process.draw_point(point=center)
process.draw_show()
while True:
    labels = np.argmin(cdist(X, center), axis=1)
    before_center = center
    center = []
    for i in range(k):
        center.append(np.mean(X[labels==i], axis=0))
    center = np.array(center)
    process.draw_label(X=X, y=labels)
    process.draw_point(point=center)
    process.draw_show()
    time.sleep(1)
    if(set([tuple(c) for c in before_center]) == set([tuple(c) for c in center])):
        break
np.save("models/training_center.npy", center)
print(center)
# 46.27631579