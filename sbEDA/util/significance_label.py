

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
