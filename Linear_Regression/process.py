import pandas as pd
import matplotlib.pyplot as plt
def read_data(path):
    data = pd.read_csv(path)
    X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
             'floors', 'condition', 'sqft_above', 'sqft_basement']].values
    Y = data[['price']].values
    # X = data[['ChieuCao']].values
    # Y = data[['CanNang']].values
    return X, Y

def draw_x(x, y):
    plt.scatter(x, y)


def draw_show():
    plt.title("Thể trạng")
    plt.xlabel("Chiều cao")
    plt.ylabel("Cân nặng")
    plt.show()

def draw_line(X, Y):
    plt.plot(X, Y, 'r')


if __name__ =="__main__":
    X, Y = read_data(path="data/the_trang_linear_regression.csv")
    draw_x(x=X, y=Y)
    draw_show()