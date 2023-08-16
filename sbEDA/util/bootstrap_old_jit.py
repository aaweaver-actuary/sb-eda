from numba import jit
import numpy as np

@jit(nopython=True)
def _jit_loop(x, y=None, stat_index='mean'):
    stat_index = stat_index.lower()
    # Mean
    if stat_index in ['mean', 'avg', 'ave', 'average', 'mu']:
        result = np.mean(x)
    # Median
    elif stat_index in ['median', 'med']:
        result = np.median(x)
    # Standard Deviation
    elif stat_index in ['std', 'stdev', 'stdeviation', 'sd', 'sigma']:
        result = np.std(x)
    # Variance
    elif stat_index in ['var', 'variance']:
        result = np.var(x)
    # Skewness and Kurtosis
    elif stat_index in ['skew', 'skewness', 'kurt', 'kurtosis']:
        m2 = m3 = m4 = 0
        mean_val = np.mean(x)
        n = len(x)
        for x1 in x:
            diff = np.subtract(x1, mean_val)
            m2 += np.power(diff, 2)
            m3 += np.power(diff, 3)
            m4 += np.power(diff, 4)
        m2 /= n
        m3 /= n
        m4 /= n
        if stat_index in ['skew', 'skewness']:
            result = m3 / m2 ** (3/2) # Skewness
        else:
            result = m4 / m2 ** 2 - 3 # Kurtosis
    # Minimum
    elif stat_index in ['min', 'minimum']:
        result = np.min(x)
    # Maximum
    elif stat_index in ['max', 'maximum']:
        result = np.max(x)
    # Range
    elif stat_index in ['range', 'rng']:
        result = np.max(x) - np.min(x)
    # IQR (Interquartile Range)
    elif stat_index in ['iqr']:
        result = np.percentile(x, 75) - np.percentile(x, 25)
    # Pearson's Correlation Coefficient
    elif stat_index in ['pearson', 'pearsonr', 'r']:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        numerator = np.sum(np.multiply(np.subtract(x, mean_x), np.subtract(y, mean_y)))
        denominator = np.sqrt(np.multiply(np.sum(np.power(np.subtract(x, mean_x), 2)), 
                                          np.sum(np.power(np.subtract(y, mean_y), 2))))
        result = numerator / denominator if denominator != 0 else 0

    # Spearman's Rank Correlation Coefficient
    elif stat_index in ['spearman', 'spearmanr', 'rho']:
        # Rank the data
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))

        # Calculate the differences between ranks
        d = np.subtract(rank_x, rank_y)

        # Calculate the square differences
        d_squared = np.power(d, 2)

        # Sum of squared differences
        sum_d_squared = np.sum(d_squared)

        # Spearman's rank correlation coefficient
        n = len(x)
        result = 1 - np.divide(6 * sum_d_squared, n * (n**2 - 1))

    # Point Biserial Correlation Coefficient
    elif stat_index in ['biserial', 'biserialr', 'rpb', 'r_biserial', 'r_pb']:
        # Assuming x is a continuous variable and y is dichotomous (0 or 1)
        mean_x = np.mean(x)
        mean_x1 = np.mean(x[y == 1]) # Mean of x for y=1
        mean_x0 = np.mean(x[y == 0]) # Mean of x for y=0
        std_x = np.std(x, ddof=0) # Standard deviation of x
        n_0 = np.sum(y == 0) # Number of cases where y=0
        n_1 = np.sum(y == 1) # Number of cases where y=1
        n = len(y) # Total number of cases

        diff_means = np.subtract(mean_x1, mean_x0)
        product_n1_n0 = np.multiply(n_1, n_0)
        div_n_squared = np.divide(product_n1_n0, np.power(n, 2))
        sqrt_proportion = np.sqrt(div_n_squared)

        result = np.multiply(np.divide(diff_means, std_x), sqrt_proportion)
    # Chi-Squared Test
    elif stat_index in ['chi2', 'chi2test', 'chi2_test']:
        # Calculate the difference between observed and expected
        difference = np.subtract(x, y)

        # Square the differences
        squared_difference = np.power(difference, 2)

        # Divide by expected, with a conditional check to avoid division by zero
        chi_square_values = np.divide(squared_difference, y, out=np.zeros_like(squared_difference), where=(y != 0))

        # Sum up the chi_square_values to get the final chi-square statistic
        result = np.sum(chi_square_values)
    # # T-Test
    # elif stat_index == 14:
    #     result = ttest_ind(x, y)[0]
    # # ANOVA
    # elif stat_index == 15:
    #     result = f_oneway(*x)[0]
    # # Information Value
    # elif stat_index == 16:
    #     result = logit(mutual_info_classif(x.reshape(-1, 1), y))
    # # Weight of Evidence
    # elif stat_index == 17:
    #     result = DescrStatsW(x, weights=y)\
    #                 .tconfint_mean()
    # # Logistic Regression Coefficient
    # elif stat_index == 18:
    #     result = DecisionTreeClassifier()\
    #                 .fit(x.reshape(-1, 1), y)\
    #                 .tree_\
    #                 .compute_feature_importances()[0]
    # # Decision Tree Feature Importance
    # elif stat_index == 19:
    #     result = DecisionTreeClassifier()\
    #                 .fit(x.reshape(-1, 1), y)\
    #                 .tree_\
    #                 .compute_feature_importances()[0]
    # # Mutual Information
    # elif stat_index == 20:
    #     result = mutual_info_classif(x.reshape(-1, 1), y)[0]
    # # Cramér's V
    # elif stat_index == 21:
    #     result = np.sqrt(chisquare(x, y)[0] / sum(x + y))
    # # Cohen's D
    # elif stat_index == 22:
    #     result = np.subtract(np.mean(x), np.mean(y)) / \
    #              np.sqrt(((len(x) - 1) * np.var(x) + (len(y) - 1) * np.var(y)) / \
    #                      (len(x) + len(y) - 2))
    # # Fisher's Exact Test
    # elif stat_index == 23:
    #     result = fisher_exact(np.array([x, y]))[0]
    # # Kendall's Tau
    # elif stat_index == 24:
    #     result = kendalltau(x, y)[0]
    # # Mann-Whitney U Test
    # elif stat_index == 25:
    #     result = mannwhitneyu(x, y)[0]



    return result

@jit(nopython=True)
def _bootstrap_samples_jit(x, y=None, stat_function='mean', n_bootstraps=1000):
    results = np.empty(n_bootstraps)
    n = len(x)
    for i in range(n_bootstraps):
        if y is None:
            sample = np.random.choice(x, n, replace=True)
        else:
            indices = np.random.choice(n, n, replace=True)
            sample_x = np.array([x[i] for i in indices])
            sample_y = np.array([y[i] for i in indices])
            results[i] = _jit_loop(x=sample_x, y=sample_y, stat_index=stat_function)
            continue
        results[i] = _jit_loop(x=sample, y=None, stat_index=stat_function)
    return results

def _bootstrap_samples(x, y=None, stat_function='mean', n_bootstraps=1000):
    try:
        # Attempt to run with JIT compilation
        return _bootstrap_samples_jit(x, y, stat_function, n_bootstraps)
    except Exception as e:
        # Fallback to pure NumPy if JIT compilation fails
        print("Note: JIT compilation failed with error: {}".format(e))
        # return np.array([
        #     stat_function(np.random.choice(data, len(data), replace=True))\
        #     for _ in range(n_bootstraps)
        #     ])

def bootstrap_ci(x, y=None, stat_function='mean', alpha=0.05, n_bootstraps=1000):
    bootstrapped_stats = _bootstrap_samples(x=x,
                                            y=y,
                                            stat_function=stat_function,
                                            n_bootstraps=n_bootstraps)
    lower = np.percentile(bootstrapped_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_stats, 100 * (1 - alpha / 2))
    return lower, upper
