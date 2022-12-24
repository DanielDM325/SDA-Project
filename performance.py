import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def precision(test_data: pd.DataFrame, prediction_data: pd.DataFrame, for_class: int = 1) -> float:
    return precision_score(test_data['fraud_bool'], prediction_data, pos_label=for_class, average='binary')


def recall(test_data: pd.DataFrame, prediction_data: pd.DataFrame, for_class: int = 1) -> float:
    return recall_score(test_data['fraud_bool'], prediction_data, pos_label=for_class, average='binary')


def f_score(precision: float, recall: float) -> float:
    return 2 / (1 / precision + 1 / recall)
