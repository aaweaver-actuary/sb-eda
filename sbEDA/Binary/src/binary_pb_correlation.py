import pandas as pd
import scipy.stats as stats

def binary_pb_correlation_test(target:pd.Series, features:pd.Series) -> float:
    """
    Calculates the point biserial correlation between a binary target
    and binary features.
    
    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Binary feature column.

    Returns
    -------
    float
        Point biserial correlation between target and features.
    """
    return stats.pointbiserialr(target, features)[0]

def binary_pb_correlation_p_value(target:pd.Series, features:pd.Series) -> float:
    """
    Calculates the point biserial correlation between a binary target
    and binary features, and returns the p-value.
    
    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Binary feature column.

    Returns
    -------
    float
        Point biserial correlation p-value between target and features.
    """
    return stats.pointbiserialr(target, features)[1]

def binary_pb_correlation_hypothesis_test(target:pd.Series,
                                          features:pd.Series,
                                          alpha_levels:list=None
                                         ) -> pd.DataFrame:
    """
    Calculates the point biserial correlation between a binary target and
    binary features, and returns a dataframe with the correlation, p-value,
    and whether the null hypothesis is rejected at a variety of alpha levels.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Binary feature column.
    alpha_levels : list, optional
        List of alpha levels to test, by default None

    Returns
    -------
    pd.DataFrame
        Point biserial correlation, p-value, and hypothesis test results.
    """
    if alpha_levels is None:
        alpha_levels = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.5]

    results = pd.DataFrame(columns=['significance_level'])
    results['significance_level'] = alpha_levels
    results['correlation'] = binary_pb_correlation_test(target, features)
    results['p_value'] = binary_pb_correlation_p_value(target, features)
    results['is_significant'] = results['p_value'] < results['significance_level']
    return results