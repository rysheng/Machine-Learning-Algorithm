import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_data():
    data = pd.read_csv('data/play_decision.csv')

    le = LabelEncoder()
    encode_saver = {}
    data['OUTLOOK'] = le.fit_transform(data['OUTLOOK'])
    encode_saver['OUTLOOK'] = le.classes_.tolist()
    data['WINDY'] = le.fit_transform(data['WINDY'])
    encode_saver['WINDY'] = le.classes_.tolist()
    data['Play'] = le.fit_transform(data['Play'])
    encode_saver['Play'] = le.classes_.tolist()

    X = data[['OUTLOOK', 'TEMPERATURE', 'HUMIDITY', 'WINDY']].values
    y = data['Play'].values
    return X, y, encode_saver

if __name__ == '__main__':
    X, y, encode_saver = get_data()
    print(encode_saver)
