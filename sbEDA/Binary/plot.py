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

_figsize = (10, 7)

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