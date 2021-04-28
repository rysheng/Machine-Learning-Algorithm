import matplotlib.pyplot as plt
import pandas as pd


def get_data(path):
    data = pd.read_csv(path)
    X = data[['ChieuCao', 'CanNang']].values
    Y = data[['Loai']].values
    return X, Y

def draw_points(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)

def draw_line(X, y):
    plt.plot(X, y)

def draw_show():
    plt.title('Perceptron Learning Algorithm')
    plt.xlabel('Chieu Cao')
    plt.ylabel('Can Nang')
    plt.show()

if __name__ == "__main__":
    X, y = get_data('data/the_trang_binary_classification.csv')
    draw_points(X=X, y=y)
    draw_show()