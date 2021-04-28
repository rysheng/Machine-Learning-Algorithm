import numpy as np
from scipy.spatial.distance import cdist

center  = np.load("models/training_center.npy")
label_meaning = ["Gầy", "Chuẩn", "Béo"]
new_human = [180, 50]
D = cdist(np.array([new_human]), center)
print(D)
human_label = np.argmin(D, axis=1)
fit_human = label_meaning[int(human_label[0])]
print("Thân hình của bạn là: " + fit_human)


