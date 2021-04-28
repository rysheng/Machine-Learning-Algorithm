import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data():
    data = pd.read_csv('data/the_trang_logistic_regression.csv')
    X = data[['TheTrang']].values
    y = data[['Loai']].values
    return X, y

def draw_Xy(X, y):
    plt.scatter(X, y, c=y)

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
    draw_show()