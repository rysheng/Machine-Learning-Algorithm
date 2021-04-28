import pandas as pd
import matplotlib.pyplot as plt

# def read_data(path):
#     data = pd.read_csv(path)
#     return data.values

def read_data(path):
    data = pd.read_csv(path)
    return data.values

def draw_x(x, y):
    plt.scatter(x, y)

def draw_label(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)

def draw_show():
    plt.title("Thể trạng")
    plt.xlabel("Chiều cao")
    plt.ylabel("Cân nặng")
    plt.show()

def draw_point(point, shape="r*"):
    plt.plot(point[:, 0], point[:, 1], shape)


if __name__ =="__main__":
    X = read_data(path="data/the_trang_kmeans.csv")
    draw_x(x=X[:, 0], y=X[:, 1])
    print(X)
    draw_show()