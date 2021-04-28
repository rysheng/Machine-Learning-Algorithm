import pandas as pd

def get_data(path):
    data = pd.read_csv(path)
    return data.values

if __name__ =="__main__":
    X = get_data('data/the_trang.csv')
    print(X)
