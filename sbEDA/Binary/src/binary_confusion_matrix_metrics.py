import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_score,
                             average_precision_score,
                             recall_score,
                             f1_score,
                             accuracy_score, # fraction of correct predictions
                             balanced_accuracy_score, # avoids inflated accuracy scores
                                                      # for imbalanced datasets
                                                      # by weighting each class equally.
                                                      # AKA "informedness"
                             roc_auc_score,
                             roc_curve,
                             fbeta_score,
                             jaccard_score,
                             hinge_loss,
                             log_loss,
                             matthews_corrcoef, 
                             precision_recall_curve, # precision-recall pairs for different probability thresholds
                             class_likelihood_ratios,# binary classification positive and negative likelihood ratios
                             hamming_loss,
                             zero_one_loss,
                             

                             )

def _validate_input(target:pd.Series = None, features:pd.Series = None) -> None:
    """
    Validates the input for binary metrics functions. Used internally, 
    not intended for external use.

    Parameters
    ----------
    target : pd.Series, optional
        Binary target column, by default None.
    features : pd.Series, optional
        Feature column, by default None.

    Raises
    ------
    AssertionError
        If target and features are not the same length.
    AssertionError
        If target is not binary.
    AssertionError
        If features is not binary.
    AssertionError
        If only one of target and features is None, or if both
        are None.
    """
    # validate input
    assert len(target) == len(features), \
        f"""Target and features must be the same length,
        but are {len(target)} and {len(features)} respectively."""

    assert set(target.unique()) == {0, 1}, \
        f"""Target must be binary, but contains {set(target.unique())}."""

    assert set(features.unique()) == {0, 1}, \
        f"""Features must be binary, but contains {set(features.unique())}."""

    assert not (target is None and features is None), \
        f"""Both target and features must be provided."""

        
def binary_true_positive(target:pd.Series,
                         features:pd.Series
                        ) -> int:
    """
    Calculates the number of true positives in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.

    Returns
    -------
    int
        Number of true positives.
    """
    # validate input
    _validate_input(target, features)

    return confusion_matrix(target, features)[1, 1]

def binary_true_negative(target:pd.Series,
                         features:pd.Series
                        ) -> int:    
    """
    Calculates the number of true negatives in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.

    Returns
    -------
    int
        Number of true negatives.
    """
    # validate input
    _validate_input(target, features)

    return confusion_matrix(target, features)[0, 0]

def binary_false_positive(target:pd.Series,
                          features:pd.Series
                         ) -> int:
    """
    Calculates the number of false positives in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.

    Returns
    -------
    int
        Number of false positives.
    """
    # validate input
    _validate_input(target, features)

    return confusion_matrix(target, features)[0, 1]

def binary_false_negative(target:pd.Series,
                          features:pd.Series
                         ) -> int:
    """
    Calculates the number of false negatives in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.

    Returns
    -------
    int
        Number of false negatives.
    """
    # validate input
    _validate_input(target, features)

    return confusion_matrix(target, features)[1, 0]

def binary_recall(target:pd.Series,
                  features:pd.Series,
                  round_to:int = None
                 ) -> float:
    """
    Calculates the recall in a binary target and binary features, defined as:
      
      true_positive / (true_positive + false_negative)

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the recall will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Recall.
    """
    # validate input
    _validate_input(target, features)

    recall = recall_score(target, features)
    tp = binary_true_positive(target, features)
    fn = binary_false_negative(target, features)

    assert np.round(recall, 6) == np.round(tp / (tp + fn), 6), \
        f"""Recall should be {tp} / ({tp} + {fn}) = {tp / (tp + fn)},
        but is {recall}."""

    if round_to is not None:
        recall = np.round(recall, round_to)

    return recall

def binary_precision(target:pd.Series,
                     features:pd.Series,
                     round_to:int = None
                    ) -> float:
    """
    Calculates the precision in a binary target and binary features, defined as:
      
      true_positive / (true_positive + false_positive)

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the precision will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Precision.
    """
    # validate input
    _validate_input(target, features)

    # calculate precision
    precision = precision_score(target, features)
    tp = binary_true_positive(target, features)
    fp = binary_false_positive(target, features)
    assert precision == tp / (tp + fp), \
        f"""Precision should be {tp} / ({tp} + {fp}) = {tp / (tp + fp)},
        but is {precision}."""

    if round_to is not None:
        precision = np.round(precision, round_to)
    return precision

def binary_f1(target:pd.Series,
              features:pd.Series,
              round_to:int = None
             ) -> float:
    """
    Calculates the F1 score in a binary target and binary features, defined as:
        
        2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the F1 score will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        F1 score.
    """
    # validate input
    _validate_input(target, features)

    # calculate f1
    f1 = f1_score(target, features)

    if round_to is not None:
        f1 = np.round(f1, round_to)
    return f1

def binary_accuracy(target:pd.Series,
                    features:pd.Series,
                    round_to:int = None
                   ) -> float:
    """
    Calculates the accuracy in a binary target and binary features, defined as:
          
          (true_positive + true_negative) /
          (true_positive + true_negative + false_positive + false_negative)

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the accuracy will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Accuracy.
    """
    # validate input
    _validate_input(target, features)

    # calculate accuracy
    accuracy = accuracy_score(target, features)
    tp = binary_true_positive(target, features)
    tn = binary_true_negative(target, features)
    fp = binary_false_positive(target, features)
    fn = binary_false_negative(target, features)

    assert accuracy == (tp + tn) / (tp + tn + fp + fn), \
        f"""Accuracy should be ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) =
        {(tp + tn) / (tp + tn + fp + fn)}, but is {accuracy}."""

    if round_to is not None:
        accuracy = np.round(accuracy, round_to)
    return accuracy

def binary_specificity(target:pd.Series,
                       features:pd.Series,
                       round_to:int = None
                      ) -> float:
    """
    Calculates the specificity in a binary target and binary features, defined as:
          
          true_negative / (true_negative + false_positive)

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the specificity will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Specificity.
    """
    # validate input
    _validate_input(target, features)

    # calculate specificity
    tn = binary_true_negative(target, features)
    fp = binary_false_positive(target, features)
    specificity = tn / (tn + fp)
    assert specificity == tn / (tn + fp), \
        f"""Specificity should be {tn} / ({tn} + {fp}) = {tn / (tn + fp)},
        but is {specificity}."""

    if round_to is not None:
        specificity = np.round(specificity, round_to)

    return specificity

def binary_roc_auc(target:pd.Series,
                   features:pd.Series,
                   round_to:int = None
                  ) -> float:
    """
    Calculates the ROC AUC in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the ROC AUC will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        ROC AUC.
    """
    # validate input
    _validate_input(target, features)

    # calculate roc auc
    roc_auc = roc_auc_score(target, features)

    if round_to is not None:
        roc_auc = np.round(roc_auc, round_to)

    return roc_auc

def binary_roc_curve(target:pd.Series,
                     features:pd.Series,
                     round_to:int = None
                     ) -> float:
    """
    Calculates the ROC curve in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the ROC curve will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        ROC curve.
    """
    # validate input
    _validate_input(target, features)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(target, features)

    if round_to is not None:
        fpr = np.round(fpr, round_to)
        tpr = np.round(tpr, round_to)
        thresholds = np.round(thresholds, round_to)

    return fpr, tpr, thresholds

def binary_mcc(target:pd.Series,
               features:pd.Series,
               round_to:int = None
               ) -> float:
    """
    Calculates the Matthews correlation coefficient for a binary target and
    binary feature. This is one of the most important metrics for binary
    classification, as well as one of the least known. It is defined as:

        (true_positive * true_negative - false_positive * false_negative) /
        sqrt((true_positive + false_positive) *
             (true_positive + false_negative) *
             (true_negative + false_positive) *
             (true_negative + false_negative))

    Note that if any of the terms in the denominator is zero, the MCC is
    undefined. In this case, the denominator is set to 1, and so the MCC
    is set to 0. This can be shown to be the correct limiting value.

    As a correlation coefficient, MCC is the geometric mean of the
    regression coefficients of the regression lines fitted to the points
    (true_positive, false_positive) and (true_negative, false_negative), as
    well as the regression coefficients of the regression lines fitted to the
    points (true_positive, false_negative) and (true_negative, false_positive).
    These are sometimes called the "correlation coefficients of the
    regression lines of the problem and its dual".

    Listed advantages of MCC over F1 score and accuracy include:
      - It works well even if the classes are of very different sizes.
      - It works well even if the number of examples in the classes is very
        unbalanced.
      - It does not assume that the examples are sampled from a population
        with a certain class balance.
      - It takes into account the balance ratios of the four confusion matrix
        categories, not just the diagonal.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the ROC curve will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Matthews correlation coefficient.
    """
    # validate input
    _validate_input(target, features)

    # calculate mcc
    tp = binary_true_positive(target, features)
    tn = binary_true_negative(target, features)
    fp = binary_false_positive(target, features)
    fn = binary_false_negative(target, features)

    mcc_numerator = tp * tn - fp * fn
    mcc_denominator2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if mcc_denominator2 == 0:
        mcc_denominator = 1
    else:
        mcc_denominator = np.sqrt(mcc_denominator2)

    mcc = mcc_numerator / mcc_denominator

    assert mcc == matthews_corrcoef(target, features), \
        f"""MCC should be {mcc_numerator} / {mcc_denominator} = {mcc},
        but sklearn.matthews_corrcoef gives
        {matthews_corrcoef(target, features)}."""

    if round_to is not None:
        mcc = np.round(mcc, round_to)

    return mcc

# average_precision_score
def binary_ave_precision(target:pd.Series,
                         features:pd.Series,
                         round_to:int = None
                         ) -> float:
    """
    Calculates the average precision in a binary target and binary features.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the average precision will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Average precision.
    """
    # validate input
    _validate_input(target, features)

    precision_recall_curve(target, features)

    # calculate average precision
    ave_precision = average_precision_score(target, features)

    if round_to is not None:
        ave_precision = np.round(ave_precision, round_to)

    return ave_precision

def binary_fowlkes_mallows_index(target: pd.Series,
                                 features: pd.Series,
                                 round_to: int = None
                                 ) -> float:
    """
    Calculates the Fowlkes-Mallows index in a binary target and binary features.

    The F-M index is calculated as the geometric mean of the precision and
    recall, with 1 being the best score and 0 being the worst score. It is
    defined as:

        TP / sqrt((TP + FP) * (TP + FN))

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the Fowlkes-Mallows index will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Fowlkes-Mallows index.
    """
    # validate input
    _validate_input(target, features)

    # calculate Fowlkes-Mallows index
    tp = binary_true_positive(target, features)
    fp = binary_false_positive(target, features)
    fn = binary_false_negative(target, features)
    
    fowlkes_mallows_index = tp / np.sqrt((tp + fp) * (tp + fn))

    if round_to is not None:
        fowlkes_mallows_index = np.round(fowlkes_mallows_index, round_to)

    return fowlkes_mallows_index

# balanced_accuracy_score
def binary_balanced_accuracy(target: pd.Series,
                             features: pd.Series,
                             round_to: int = None
                             ) -> float:
    """
    Calculates the balanced accuracy in a binary target and binary features.

    The balanced accuracy is calculated as the arithmetic mean of the
    sensitivity and specificity, with 1 being the best score and 0 being the
    worst score. It is defined as:

        (TP / (TP + FN) + TN / (TN + FP)) / 2

    The balanced accuracy is also known as the average accuracy. It is used to
    evaluate binary classification problems which have an unequal number of
    observations in each class and where TP and TN are more important than FP
    and FN.
    
    It is useful when there is high class imbalance.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the balanced accuracy will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Balanced accuracy.
    """
    # validate input
    _validate_input(target, features)

    # calculate balanced accuracy
    tp = binary_true_positive(target, features)
    tn = binary_true_negative(target, features)
    fp = binary_false_positive(target, features)
    fn = binary_false_negative(target, features)

    balanced_accuracy = balanced_accuracy_score(target, features)
    balanced_accuracy_check = (tp / (tp + fn) + tn / (tn + fp)) / 2

    # test that the two methods give the same result
    assert balanced_accuracy == balanced_accuracy_check, \
        f"""Balanced accuracy should be
        ({tp} / ({tp} + {fn}) + {tn} / ({tn} + {fp})) / 2 = {balanced_accuracy_check},
        but sklearn.metrics.balanced_accuracy_score gives
        {balanced_accuracy}."""
    
    if round_to is not None:
        balanced_accuracy = np.round(balanced_accuracy, round_to)

    return balanced_accuracy

# fbeta_score,
def binary_fbeta_score(target: pd.Series,
                       features: pd.Series,
                       beta: float = 1,
                       round_to: int = None
                       ) -> float:
    """
    Calculates the F-beta score in a binary target and binary features.

    The F-beta score is the weighted harmonic mean of the precision and recall,
    with 1 being the best score and 0 being the worst score. It is defined as:

        (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    The `beta` parameter determines the weight of recall precision relative to
    recall. beta < 1 gives more weight to recall, beta > 1 gives more weight to
    precision, and beta = 1 gives equal weight to both precision and recall.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    beta : float, optional
        The beta parameter determines the weight of the precision in the
        combined score. beta < 1 gives more weight to recall, beta > 1 gives
        more weight to precision, and beta = 1 gives equal weight to both
        precision and recall, by default 1
    round_to : int, optional
        If not None, the F-beta score will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        F-beta score.
    """
    # validate input
    _validate_input(target, features)

    # calculate F-beta score
    fbeta = fbeta_score(target, features, beta=beta)

    # calculate F-beta score manually
    tp = binary_true_positive(target, features)
    fp = binary_false_positive(target, features)
    fn = binary_false_negative(target, features)
    beta2 = beta**2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fbeta_check = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)

    # test that the two methods give the same result
    assert fbeta == fbeta_check, \
        f"""F-beta score should be
(1 + beta^2) * (precision * recall) / (beta^2 * precision + recall) = 
(1 + {beta}^2) * ({precision} * {recall}) / ({beta}^2 * {precision} + {recall}) = \
{fbeta_check}"""

    # round if requested
    if round_to is not None:
        fbeta = np.round(fbeta, round_to)

    return fbeta

#  jaccard_score,
def binary_jaccard_score(target: pd.Series,
                         features: pd.Series,
                         round_to: int = None
                         ) -> float:
    """
    Calculates the Jaccard score in a binary target and binary features.

    The Jaccard score is the intersection over union of the predicted and
    actual values, with 1 being the best score and 0 being the worst score. It
    is defined as:

        TP / (TP + FP + FN)

    The Jaccard score is also known as the Jaccard similarity coefficient and
    the Jaccard index. It is used to evaluate similarity between two sets.
    
    It is particularly useful when you want to understand how well the positive
    class is predicted.

    Parameters
    ----------
    target : pd.Series
        Binary target column.
    features : pd.Series
        Feature column.
    round_to : int, optional
        If not None, the Jaccard score will be rounded to this number
        of decimals, by default None

    Returns
    -------
    float
        Jaccard score.
    """



#  hinge_loss,
#  log_loss,
#  precision_recall_curve, # precision-recall pairs for different probability thresholds
#  class_likelihood_ratios,# binary classification positive and negative likelihood ratios
#  hamming_loss,
#  zero_one_loss

