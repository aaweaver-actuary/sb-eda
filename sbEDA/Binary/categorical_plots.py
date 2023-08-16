import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from util.significance_label import _get_significance_band

_figsize = (10, 7)

def _plot_label(s: str):
    s = s.replace('_', ' ').title()
    return f"[{s}]" if s[0] == '[' else s

def plot_bar_chart(feature,
                   target,
                   figsize=_figsize,
                   ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=feature,
                  hue=target,
                  ax=ax)
    
    # set x and y labels
    ax.set_xlabel(_plot_label(feature.name))
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of [{_plot_label(feature.name)}] by \
[{_plot_label(target.name)}]")
    return ax

def plot_stacked_bar_chart(feature,
                           target,
                           x_offset=0.05,
                           y_offset=0.0225,
                           figsize=_figsize,
                           ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ct = pd.crosstab(feature, target, normalize='index')
    bars = ct.plot.bar(stacked=True, ax=ax)
    
    # Adding horizontal dashed lines
    for container in bars.containers:
        for bar in container.patches:
            y = bar.get_y() + bar.get_height()
            x_value = bar.get_x() - x_offset  
            ax.axhline(y,
                       xmin=0,
                       xmax=bar.get_x() + bar.get_width(),
                       linestyle='--',
                       color='black')
            ax.annotate(f"{y*100:.1f}%",
                        xy=(x_value, y + y_offset),
                        textcoords='data',
                        va='center',
                        ha='right')
    
    # Set x and y labels
    ax.set_xlabel(_plot_label(feature.name))
    ax.set_ylabel("Count")

    # Set y ticks to percentage
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(y*100)}%" for y in yticks])

    # Set title
    ax.set_title(f"Distribution of [{_plot_label(feature.name)}] by \
[{_plot_label(target.name)}]")
    return ax

def plot_chi_squared_test(feature,
                          target,
                          alpha=0.05,
                          figsize=_figsize,
                          ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ct = pd.crosstab(feature, target)
    chi2, p, _, _ = chi2_contingency(ct)
    chi2_results_msg = f"Chi-Squared Test: chi2 = {chi2:.4f}, p-value = {p:.4f}"
    significance_message = f"{_get_significance_band(p, 'indicated association')}"
    sns.heatmap(ct, annot=True, fmt=',', cmap='viridis', ax=ax)
    ax.set_title(f"Chi-Squared Test for Independence\n{chi2_results_msg}\n\
{significance_message}")
    return ax

def plot_point_plot(feature,
                    target,
                    figsize=_figsize,
                    ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.pointplot(x=feature, y=target, errorbar=('ci', 95), ax=ax)  # Including 95% confidence intervals
    sns.stripplot(x=feature, y=target, color='gray', alpha=0.5, ax=ax) # Including strip plot

    # Set title and labels
    ax.set_title(f"Relationship of [{_plot_label(feature.name)}] with [{_plot_label(target.name)}] (95% CI)")
    ax.set_xlabel(_plot_label(feature.name))
    ax.set_ylabel(_plot_label(target.name))
    
    return ax


def categorical_plots(feature, target, figsize=(14, 14)):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Plotting the Bar Chart
    ax1 = plot_bar_chart(feature, target, ax=ax1)

    # Plotting the Stacked Bar Chart
    ax2 = plot_stacked_bar_chart(feature, target, ax=ax2)

    # Plotting the Chi-Squared Test
    ax3 = plot_chi_squared_test(feature, target, ax=ax3)

    # Plotting the Point Plot
    ax4 = plot_point_plot(feature, target, ax=ax4)

    plt.tight_layout()
    plt.show()
