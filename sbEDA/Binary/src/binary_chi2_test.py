import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def binary_chi2_test(target:pd.Series, features:pd.Series) -> float:
    """
    Calculates the chi2 test between a binary target and binary features.
    
    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Binary feature column.

    Returns
    -------
    float
        Chi2 test between target and features.
    """
    return np.round(chi2_contingency(pd.crosstab(target, features))[0], 4)

def binary_chi2_p_value(target:pd.Series, features:pd.Series) -> float:
    """
    Calculates the chi2 test between a binary target and binary features,
    and returns the p-value.
    
    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Binary feature column.

    Returns
    -------
    float
        Chi2 test p-value between target and features.
    """
    return np.round(chi2_contingency(pd.crosstab(target, features))[1], 4)

def binary_chi2_hypothesis_test(target:pd.Series, features:pd.Series, alpha_levels:list = None) -> float:
    """
    Calculates the chi2 test between a binary target and binary features,
    and returns a dataframe with the chi2 test, p-value, and whether the null
    hypothesis is rejected at a variety of alpha levels.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Binary feature column.
    alpha_levels : list, optional
        List of alpha levels to test, by default None, which uses a default set of
        alpha levels.

    Returns
    -------
    pd.DataFrame  

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sbEDA.Binary.src.binary_chi2_test import binary_chi2_hypothesis_test
    >>> np.random.seed(42)
    >>> target = pd.Series(np.random.randint(0, 2, 100))
    >>> features = pd.Series(np.random.randint(0, 2, 100))
    >>> binary_chi2_hypothesis_test(target, features)
    """
    if alpha_levels is None:
        alpha_levels = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.5]

    test_statistic = binary_chi2_test(target, features)
    p_value = binary_chi2_p_value(target, features)

    results = pd.DataFrame(columns=['significance_level'])
    results['significance_level'] = alpha_levels
    results['chi_2_test_statistic'] = test_statistic
    results['p_value'] = p_value
    results['reject_null'] = results['p_value'] < results['significance_level']

    return results