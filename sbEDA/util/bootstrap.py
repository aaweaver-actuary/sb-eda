import pandas as pd
import numpy as np

def bootstrap(x,
              y=None,
              stat_function=None,
              n_bootstraps=1000,
              subset=None):
    """
    Calculate bootstrap samples for a given statistic.

    Parameters
    ----------
    x : array-like
        Data to be bootstrapped.
    y : array-like, optional
        Second data to be bootstrapped, depending on the statistic.
    stat_function : function, optional
        Statistic to be calculated. If None, the mean is calculated.
    n_bootstraps : int, optional
        Number of bootstrap samples to be generated. Default is 1000.
    subset : float, optional
        Size of the subsample to be drawn from the data. If None, the size of
        the subsample is the same as the size of the data. Must be between 0
        and 1. Default is None.

    Returns
    -------
    bootstrapped_stats : array-like

    Example Usage
    -------------
    >>> import numpy as np
    >>> from sbEDA.util.bootstrap import bootstrap
    >>> x = np.random.normal(0, 1, 1000)

    >>> # test that the shape of the bootstrapped means is (1000,)
    >>> bootstrapped_stats = bootstrap(x=x, stat_function=np.mean)
    >>> bootstrapped_stats.shape
    (1000,)

    >>> # test that the mean of the bootstrapped means is close to the mean of x
    >>> np.round(bootstrap(x=x, stat_function=np.mean).mean(), 1)
    0.0

    >>> # get the bootstrap samples for the pearson correlation coefficient of x and y
    >>> y = np.random.normal(0, 1, 1000)
    >>> bootstrap(x=x, y=y, stat_function=np.corrcoef).shape
    (1000, 2, 2)

    >>> # first item is the first correlation matrix between the first 
    >>> # bootstrapped sample of x and y
    >>> bootstrap(x=x, y=y, stat_function=np.corrcoef)[0]
    array([[1.        , 0.00306492],
            [0.00306492, 1.        ]])
    """
    assert n_bootstraps > 0, f"n_bootstraps: {n_bootstraps} must be > 0"
    assert subset is None or (subset > 0 and subset <= 1), \
        f"subset: {subset} must be None or between 0 and 1"
    assert y is None or len(x) == len(y), \
        f"len(x): {len(x)} must be equal to len(y): {len(y)}"
    
    # get the number of data points
    n = len(x)
    if subset is None:
        subset_size = n
    else:
        subset_size = int(subset * n)

    # if no statistic function is provided, use mean
    if stat_function is None:
        stat_function = np.mean

    bootstrapped_stats = []
    for _ in range(n_bootstraps):
        # sample with replacement
        boot_indices = np.random.choice(n, subset_size, replace=True)
        boot_x = x[boot_indices]
        if y is not None:
            boot_y = y[boot_indices]
            bootstrapped_stats.append(stat_function(boot_x, boot_y))
        else:
            bootstrapped_stats.append(stat_function(boot_x))
    return pd.Series(bootstrapped_stats)

def bootstrap_ci(x,
                 y=None,
                 stat_function=None,
                 alpha=0.05,
                 n_bootstraps=1000):
    """
    Calculate bootstrap confidence interval for a given statistic.

    Parameters
    ----------
    x : array-like
        Data to be bootstrapped.
    y : array-like, optional
        Second data to be bootstrapped, depending on the statistic.
    stat_function : function, optional
        Statistic to be calculated. If None, the mean is calculated.
    alpha : float, optional
        Significance level. Default is 0.05.
    n_bootstraps : int, optional
        Number of bootstrap samples to be generated. Default is 1000.

    Returns
    -------
    lower : float
    """
    bootstrapped_stats = bootstrap(x=x,
                                   y=y,
                                   stat_function=stat_function,
                                   n_bootstraps=n_bootstraps)
    lower = bootstrapped_stats.quantile(alpha / 2)
    upper = bootstrapped_stats.quantile(1 - alpha / 2)
    return lower, upper
