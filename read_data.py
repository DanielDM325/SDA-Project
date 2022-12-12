import pandas as pd


def read_dataset(path: str, one_hot_encoding=False, drop_first=True) -> pd.DataFrame:
    data = pd.read_csv(path, encoding='utf-8')
    if one_hot_encoding:
        data_one_hot_encoded = pd.get_dummies(data, drop_first=drop_first)
        return data_one_hot_encoded
    else:
        return data
