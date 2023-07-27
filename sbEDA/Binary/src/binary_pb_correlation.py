import pandas as pd
import scipy.stats as stats

def binary_pb_correlation(target:pd.Series, features:pd.Series) -> float:
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
    return stats.pointbiserialr(target, features)