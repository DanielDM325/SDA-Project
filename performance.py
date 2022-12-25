import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def precision(test_data: pd.DataFrame, prediction_data: pd.DataFrame, for_class: int = 1) -> float:
    """
    Calculates the classification precion score
    https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
    precision = true positives / (true positives / false positives)
    """
    return precision_score(test_data['fraud_bool'], prediction_data, pos_label=for_class, average='binary')


def recall(test_data: pd.DataFrame, prediction_data: pd.DataFrame, for_class: int = 1) -> float:
    """
    Calculates the classification recall score
    https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
    recall = true positives / (true positives / false negatives)
    """
    return recall_score(test_data['fraud_bool'], prediction_data, pos_label=for_class, average='binary')


def f_score(precision: float, recall: float) -> float:
    return 2 / (1 / precision + 1 / recall)
