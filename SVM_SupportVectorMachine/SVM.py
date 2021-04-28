from sklearn.preprocessing import MinMaxScaler
from SVM_SupportVectorMachine.reprocess import *

X, y = get_data()
# draw_Xy(X, y)
# draw_show(title='SVM', xlabel='Chiều cao', ylabel='Cân nặng')

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

X_scaler = min_max_scaler.fit_transform(X)
scaler_saved = [min_max_scaler.min_, min_max_scaler.scale_]
draw_Xy(X_scaler, y)
draw_show(title='SVM', xlabel='Chiều cao', ylabel='Cân nặng')


C = 100
lamda = 1./C
w = .1 * np.random.randn(X_scaler.shape[1])
b = np.random.randn()


# loss function
def get_loss(X, y, w, b):
    loss = (np.sum(np.maximum(0, 1 - y * (X.dot(w) + b))) + .5 * lamda * w.dot(w)) / X.shape[0]
    return loss

# optimze function
def grad(X, y, w, b):
    yz = y * (X.dot(w) + b)
    _yX = -X*y[:, np.newaxis]
    grad_w = (np.sum(_yX[yz <= 1], axis=0) + lamda * w) / X.shape[0]
    grab_b = (-np.sum(y[yz <= 1])) / X.shape[0]
    return grad_w, grab_b


epochs = 10000
learning_rate = 0.01

for ep in range(epochs):
    gw, gb = grad(X_scaler, y, w, b)
    w -= learning_rate * gw
    b -= learning_rate * gb
    if (ep % 1000 == 0):
        loss_val = get_loss(X_scaler, y, w, b)
        print(f'iter {ep}| loss: {loss_val}')

# w0*x + w1*y + b = 0
x_draw = np.array([-1, 1])
y_draw = (-b - w[0] * x_draw) / w[1]

def draw_line(x, y):
    plt.plot(x, y, 'g')

draw_Xy(X_scaler, y)
draw_line(x_draw, y_draw)
draw_show(title='SVM', xlabel='Chiều cao', ylabel='Cân nặng')