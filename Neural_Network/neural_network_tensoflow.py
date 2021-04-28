import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from Neural_Network import reprocess
from sklearn.preprocessing import MinMaxScaler

X, y = reprocess.get_data()
min_max_scaler = MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)

model = Sequential([
    Dense(units=20, activation='relu', input_shape=(2,)),
    Dense(units=6, activation='softmax'),
])

loss = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='SGD', loss=loss, metrics=['accuracy'])


hist = model.fit(X_scaler, y, epochs=1000)
print(hist)
losses_hist = hist.history['loss']
reprocess.draw_line(list(range(len(losses_hist))), losses_hist)

acc_hist = hist.history['accuracy']
reprocess.draw_line(list(range(len(acc_hist))), acc_hist)