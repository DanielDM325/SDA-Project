import pandas as pd


def read_dataset(path: str, process=False, drop_first=True) -> pd.DataFrame:
    data = pd.read_csv(path, encoding='utf-8')
    if process:
        processed = pd.get_dummies(data, drop_first=drop_first)
        irrelevant_columns = [column for column in processed.columns if processed[column].nunique() == 1]
        processed.drop(irrelevant_columns, axis=1, inplace=True)
        return processed
    else:
        return data
