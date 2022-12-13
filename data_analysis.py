from scipy import stats
import numpy as np
import pandas as pd


def kolmogorov_smirnov_similarity(sample_1, sample_2) -> float:
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        ks_statistic = stats.ks_2samp(sample_1, sample_2)[0]
        return ks_statistic
    elif type(sample_1) == pd.DataFrame and type(sample_2) == pd.DataFrame:
        if list(sample_1.columns.values) != list(sample_2.columns.values):
            return None
        ks_statistics = list()
        for column in sample_1.columns:
            ks_statistics.append(stats.ks_2samp(sample_1[column].values, sample_2[column].values)[0])
        return np.array(ks_statistics)
    else:
        return None


def epps_singleton_similarity(sample_1, sample_2):
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        ep_statistic = stats.epps_singleton_2samp(sample_1, sample_2)[0]
        return ep_statistic
    elif type(sample_1) == pd.DataFrame and type(sample_2) == pd.DataFrame:
        if list(sample_1.columns.values) != list(sample_2.columns.values):
            return None
        ep_statistics = list()
        for column in sample_1.columns:
            ep_statistics.append(stats.epps_singleton_2samp(sample_1[column].values, sample_2[column].values)[0])
        return np.array(ep_statistics)
    else:
        return None
