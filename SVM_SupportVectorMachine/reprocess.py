import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_data():
    data = pd.read_csv('data/the_trang_binary_classification.csv')
    X = data[['ChieuCao', 'CanNang']].values
    y = data['Loai'].values
    return X, y

def draw_Xy(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)

def draw_show(title='', xlabel='', ylabel=''):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':
    X, y = get_data()
    draw_Xy(X, y)
    draw_show(title='SVM', xlabel='Chiều cao', ylabel='Cân nặng')