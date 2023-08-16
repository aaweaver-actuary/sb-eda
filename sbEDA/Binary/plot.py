"""
This module contains the functions for plotting the results of the exploratory
data analysis on a binary classification problem.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from scipy.stats import ks_2samp

_figsize = (10, 7)

def _get_significance_band(p_value, statistic):
    if p_value < 0.01:
        significance_statement = f"Extremely likely that the {statistic} is significant"
    elif p_value < 0.05:
        significance_statement = f"Very likely that the {statistic} is significant"
    elif p_value < 0.10:
        significance_statement = f"Somewhat likely that the {statistic} is significant"
    else:
        significance_statement = f"Unlikely that the {statistic} is significant"
    return significance_statement

def _format_plot(feature,
                 target,
                 ax=None,
                 n_series=2,
                 title=None,
                 legend=True):
    """
    Format the plot.
    
    Parameters
    ----------
    feature : str
        The name of the feature to plot.
    target : str
        The name of the target variable.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes to plot on.
    title : str, optional
        The title of the plot. The keywords "feature" and "target" will be
        replaced with the actual feature and target names.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    # get the axes
    if ax is None:
        ax = plt.gca()

    # get the feature and target names and title-case them
    feature_t = feature.replace("_", " ").title()
    target_t = target.replace("_", " ").title()

    feature_t = f"[{feature_t}]" if "[" not in feature_t else feature_t
    target_t = f"[{target_t}]" if "[" not in target_t else target_t
    
    # set the plot title, replacing the strings "feature" and "target" with
    # the actual feature and target name variables `feature_t` and `target_t`
    real_title = title.replace("feature", feature_t).replace("target", target_t)
    ax.set_title(real_title)
    
    if n_series==2:
        # set the y-axis label
        ax.set_ylabel(feature_t)
        
        # set the x-axis label
        ax.set_xlabel(target_t)
    elif n_series==1:
        # set the y-axis label
        ax.set_ylabel("Density")
        
        # set the x-axis label
        ax.set_xlabel(feature_t)

    
    if legend:
        # set the labels for the legend
        ax.legend(title=target_t)
    
    return ax

def binary_distribution_plot(feature,
                             target,
                             df,
                             ax=None,
                             title="Distribution of feature by target",
                             figsize=None,
                             save_fig=False,
                             save_path=None,
                             **kwargs):
    """
    Plot the distribution of the feature split by the target variable.

    Parameters
    ----------
    feature : str
        The name of the feature to plot.
    target : str
        The name of the target variable.
    df : pandas.DataFrame
        The dataframe containing the feature and target variable.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes to plot on.
    title : str, optional
        The title of the plot. The keywords "feature" and "target" will be
        replaced with the actual feature and target names.
        It is done this way to keep the function as general as possible.
    figsize : tuple, optional
        The size of the figure. If None, the default of (15, 8) is used.
    **kwargs : dict
        Keyword arguments to pass to seaborn.histplot.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the distribution plot.
    """
    # get the figure size
    if figsize is None:
        figsize = _figsize

    # get the axes
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)        

    # plot the distribution
    sns.histplot(data=df,
                    x=feature,
                    hue=target,
                    ax=ax,
                    kde=True,
                    fill=True,
                    stat="density",
                    **kwargs)
    
    # format the plot
    ax = _format_plot(feature, target, ax=ax, title=title, n_series=1, legend=False)

    if save_fig:
        if save_path is None:
            save_path = "plots"
        plt.savefig(f"{save_path}/{feature}_distribution.png",
                    dpi=300,
                    bbox_inches="tight")
        

    return ax
    




def binary_boxplot(feature,
                   target,
                   df,
                   ax=None,
                   title="Binary Boxplot of feature by target",
                   figsize=None,
                   save_fig=False,
                   save_path=None,
                   **kwargs):
    """
    Plot a boxplot of the feature split by the target variable.
    
    Parameters
    ----------
    feature : str
        The name of the feature to plot.
    target : str
        The name of the target variable.
    df : pandas.DataFrame
        The dataframe containing the feature and target variable.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes to plot on.
    title : str, optional
        The title of the plot. The keywords "feature" and "target" will be
        replaced with the actual feature and target names.
        It is done this way to keep the function as general as possible.
    figsize : tuple, optional
        The size of the figure. If None, the default of (15, 8) is used.
    **kwargs : dict
        Keyword arguments to pass to seaborn.boxplot.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the boxplot.
    """
    # get the figure size
    if figsize is None:
        figsize = _figsize

    # get the axes
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # plot the boxplot
    sns.boxplot(x=target,
                y=feature,
                data=df,
                ax=ax,
                **kwargs)
    
    ax = _format_plot(feature, target, ax=ax, title=title, n_series=2)

    if save_fig:
        if save_path is None:
            save_path = "plots"
        plt.savefig(f"{save_path}/{feature}_distribution.png",
                    dpi=300,
                    bbox_inches="tight")

    return ax

def binary_roc_auc(feature, target, alpha=0.05, figsize=_figsize, save_fig=False, save_path=None):



    # Reshape feature for scikit-learn
    feature = feature.values.reshape(-1, 1)

    # Fit logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(feature, target)

    # Predict probability
    probas = log_reg.predict_proba(feature)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(target, probas)
    roc_auc = auc(fpr, tpr)

    # Fit logistic regression model with statsmodels for p-value
    feature_with_const = sm.add_constant(feature)
    logit_model = sm.Logit(target, feature_with_const).fit(disp=0)
    p_value = logit_model.pvalues[1]

    # get significance band
    s_band = _get_significance_band(p_value, 'coefficient')

    # Plot ROC curve
    plt.figure(figsize=_figsize)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic from \
Single-Variable Logistic Regression\n\
[Coefficient]={logit_model.params[1]:.3f}, \
[p-value]={p_value:0.4f}, \
{s_band}')
    plt.legend(loc='lower right')

    # Annotate plot with p-value and other details if needed
    significance = "Significant" if p_value < alpha else "Not Significant"
    annotation = f"p-value: {p_value:.3f}\n{significance}\nCoefficient: {logit_model.params[1]:.3f}"
    plt.annotate(annotation, xy=(0.6, 0.2), xycoords='axes fraction')

    plt.show()


def plot_ks_statistic(feature,
                      target,
                      alpha=0.05,
                      figsize=_figsize,
                      save_fig=False,
                      save_path=None):
    # Create figure
    plt.figure(figsize=figsize)    

    # Separate feature by target classes
    class_0 = feature[target == 0]
    class_1 = feature[target == 1]

    # Compute empirical CDF for each class
    def cdf_0(x):
        return (class_0 <= x).sum() / len(class_0)
    def cdf_1(x):
        return (class_1 <= x).sum() / len(class_1)

    # Compute K-S statistic
    ks_stat, p_value = ks_2samp(class_0, class_1)

    # Create range for x-axis
    x_range = np.linspace(feature.min(), feature.max(), 1000)

    # Get significance band
    s_band = _get_significance_band(p_value, 'difference in CDFs')

    # Compute significance level & get statement
    if p_value < alpha:
        significance_statement = f"Significant at the {alpha:.1%} level"
        significance_statement2 = f"Significant until the {p_value:.5%} level"
    else:
        significance_statement = f"Not significant at the {alpha:.1%} level"
        significance_statement2 = significance_statement

    # Plot CDFs
    plt.plot(x_range,
             [cdf_0(x) for x in x_range],
             label='class 0 CDF'.replace("class", target.name).replace("_", " ").title(),
             color='navy')
    plt.plot(x_range,
             [cdf_1(x) for x in x_range],
             label='class 1 CDF'.replace("class", target.name).replace("_", " ").title(),
             color='darkorange')
    plt.xlabel('feature Value'.replace("feature", feature.name).replace("_", " ").title())
    plt.ylabel('CDF')
    plt.title(f'Kolmogorov-Smirnov Plot\n{s_band}')
    plt.legend()

    # Annotate with K-S statistic and p-value
    annotation = f"K-S Statistic: {ks_stat:.3f}\np-value: {p_value:.3f}\n"
    annotation += significance_statement
    if significance_statement != significance_statement2:
        annotation += f"\n{significance_statement2}"
    plt.annotate(annotation, xy=(0.6, 0.2), xycoords='axes fraction')

    if save_fig:
        if save_path is None:
            save_path = "plots"
        plt.savefig(f"{save_path}/{feature}_ks_statistic.png",
                    dpi=300,
                    bbox_inches="tight")

    plt.show()

