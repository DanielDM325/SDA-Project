from scipy import stats
import numpy as np
import pandas as pd


def kolmogorov_smirnov_similarity(sample_1, sample_2):
    """
    Uses the Kolmogorov-Smirnov test to calculate similarity between two data
    samples. This function either accepts 2 numpy arrays for which it will
    calculate the KS statistic and p-value or two pandas DataFrames with the
    same columns for which it will calculate the KS statistic and p-value for
    every column. Look up
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
    to understand the KS statistic and p-value.
    """
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        ks_statistic = stats.ks_2samp(sample_1, sample_2)
        return ks_statistic
    elif type(sample_1) == pd.DataFrame and type(sample_2) == pd.DataFrame:
        if list(sample_1.columns.values) != list(sample_2.columns.values):
            return None
        ks_statistics = list()
        for column in sample_1.columns:
            ks_statistics.append(stats.ks_2samp(sample_1[column].values,
                                                sample_2[column].values))
        return np.array(ks_statistics)
    else:
        return None


def epps_singleton_similarity(sample_1, sample_2):
    """
    Uses the Epps-Singleton test to calculate similarity between two data
    samples. This function either accepts 2 numpy arrays for which it will
    calculate the ES statistic and p-value or two pandas DataFrames with the
    same columns for which it will calculate the ES statistic and p-value for
    every column. Look up
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.epps_singleton_2samp.html
    to understand the KS statistic and p-value.
    """
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        ep_statistic = stats.epps_singleton_2samp(sample_1, sample_2)
        return ep_statistic
    elif type(sample_1) == pd.DataFrame and type(sample_2) == pd.DataFrame:
        if list(sample_1.columns.values) != list(sample_2.columns.values):
            return None
        ep_statistics = list()
        for column in sample_1.columns:
            try:
                ep_statistics.append(stats.epps_singleton_2samp(sample_1[column].values,
                                                                sample_2[column].values))
            except np.linalg.LinAlgError:
                ep_statistics.append([-1, -1])
        return np.array(ep_statistics)
    else:
        return None


def cross_correlate(a, v):
    """
    Calculates the cross correlation for a and v. This will serve as a
    function for the pandas corr method as argument.
    """
    return np.correlate(a, v)[0]


def cross_correlation_coefficient(sample_1, sample_2=None):
    """
    Calculates the cross correlation. Either inbetween 2 numpy arrays or for
    all columns in a DataFrame.
    """
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        correlation = np.correlate(sample_1, sample_2)
        return correlation
    elif type(sample_1) == pd.DataFrame:
        return sample_1.corr(cross_correlate)
    else:
        return None


def pearson_correlation_coefficient(sample_1, sample_2=None):
    """
    Calculates the Pearson correlation. Either inbetween 2 numpy arrays or for
    all columns in a DataFrame.
    """
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        correlation = stats.pearsonr(sample_1, sample_2)
        return correlation
    elif type(sample_1) == pd.DataFrame:
        return sample_1.corr('pearson')
    else:
        return None


def spearman_correlation_coefficient(sample_1, sample_2=None):
    """
    Calculates the Spearman correlation. Either inbetween 2 numpy arrays or
    for all columns in a DataFrame.
    """
    if type(sample_1) == np.ndarray and type(sample_2) == np.ndarray:
        correlation = stats.spearmanr(sample_1, sample_2)
        return correlation
    elif type(sample_1) == pd.DataFrame:
        return sample_1.corr('spearman')
    else:
        return None


def bootstrap_mean_standard_deviation(sample, sub_sample, size: int, iterations: int = 10000, columns_include=None, columns_exclude=None):
    """
    A bootstrap procedure to test the mean and standard deviation of a
    subsample in relation to a bigger sample of size size. Specific columns
    can be either included or excluded. If none provided all columns will be
    included. Number of iterations can be provided for the amount of bootstrap
    samples generated. It will calculate p-values and also return bootstrapped
    means and standard deviations so a histogram can be plotted. The p-value
    is two-tailed.
    """
    statistics = list()
    for column in sample.loc[:, ~sample.columns.isin(columns_exclude)].columns if columns_exclude else columns_include if columns_include else sample.columns:
        main_mean = sub_sample[column].mean()
        main_standard_deviations = sub_sample[column].std()
        means = list()
        standard_deviations = list()
        for _ in range(iterations):
            bootstrap_sample = sample[column].sample(n=size, replace=True)
            means.append(bootstrap_sample.mean())
            standard_deviations.append(bootstrap_sample.std())
        means = np.array(means)
        standard_deviations = np.array(standard_deviations)
        smaller_mean = True if main_mean < means.mean() else False
        smaller_standard_deviation = True if main_standard_deviations < standard_deviations.mean() else False
        p_value_mean = ((np.count_nonzero(means < main_mean) if smaller_mean else np.count_nonzero(means > main_mean)) / iterations) * 2
        p_value_standard_deviation = ((np.count_nonzero(standard_deviations < main_standard_deviations) if smaller_standard_deviation else np.count_nonzero(standard_deviations > main_standard_deviations)) / iterations) * 2
        statistics.append((column, means, standard_deviations, p_value_mean, p_value_standard_deviation))
    return statistics


def find_distribution(sample, distributions_consider: list = list()) -> tuple:
    """
    Will find the best possible distribution type and its parameters using
    Maximum Likelihood Estimation. The goodness of fit is calculated using the
    Kolmogorov-Smirnov test. distributions_consider can be provided with a
    list of distrbutions (the names of scipy continuous distubtions in string
    format will suffice:
    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
    If none provided the distributions below in distribution_names will be
    considered. It must be noted it can take a lot of computation if the list
    is long. The distribution name and its parameters and the KS-statistic and
    its p-value are returned.
    """
    distribution_names = ['arcsine', 'alpha', 'beta', 'cosine', 'gamma',
                          'pareto', 'rayleigh', 'norm', 'lognorm', 'expon',
                          'dweibull']
    if distributions_consider:
        distribution_names = distributions_consider
    mean = np.mean(sample)
    standard_deviation = np.std(sample)
    best_parameters = getattr(stats, distribution_names[0]).fit(sample,
                                                                loc=mean,
                                                                scale=standard_deviation)
    best_ks_statistic = stats.kstest(sample, distribution_names[0],
                                     args=best_parameters)
    best_distribition = 0
    for d, distribution in enumerate(distribution_names[1:]):
        parameters = getattr(stats, distribution).fit(sample, loc=mean,
                                                      scale=standard_deviation)
        ks_statistic = stats.kstest(sample, distribution_names[d + 1],
                                    args=parameters)
        if ks_statistic[0] < best_ks_statistic[0]:
            best_ks_statistic = ks_statistic
            best_parameters = parameters
            best_distribition = d + 1
    return distribution_names[best_distribition], best_parameters, best_ks_statistic
