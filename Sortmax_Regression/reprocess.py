import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data():
    data = pd.read_csv('data/the_trang.csv')
    X = data[['ChieuCao', 'CanNang']].values
    y = data[['Loai']].values
    return X, y

def draw_Xy(X, y):
    plt.scatter(X[:, 0], X[:, -1], c=y)

def draw_show(title='', xlabel='', ylabel=''):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def draw_line(X, y):
    plt.plot(X, y)

if __name__ == "__main__":
    X, y = get_data()
    draw_Xy(X, y)
    print(type(X))
    print(X)
    draw_show()