import pandas as pd
import numpy as np
from sklearn.metrics import precision_score


def precision(test_data: pd.DataFrame, prediction_data: pd.DataFrame, for_class: int = 1) -> float:
    return precision_score(test_data['fraud_bool'], prediction_data, pos_label=for_class, average='binary')

