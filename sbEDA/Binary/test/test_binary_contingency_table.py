import pandas as pd

def test_binary_contingency_table(target: pd.Series, feature: pd.Series) -> pd.DataFrame:
    """
    Returns a binary contingency table for a given target and feature.

    Parameters
    ----------
    target : pd.Series
        A binary target.
    feature : pd.Series
        A binary feature.

    Returns
    -------
    pd.DataFrame
        A binary contingency table.
    """
    return pd.crosstab(target, feature)
