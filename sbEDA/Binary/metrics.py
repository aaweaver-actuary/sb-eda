from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# load convergence warnings so they can get ignored
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import numpy as np
import statsmodels.api as sm
import pandas as pd

def evaluate_feature(feature: pd.Series,
                     target: pd.Series,
                     n_splits: int = 5,
                     alpha: float = 0.05):
    """
    Evaluates a feature by fitting a logistic regression model to the feature
    and target. The evaluation is done using k-fold cross validation. The
    evaluation metrics are the average mean squared error, average coefficient
    of the feature, and the number of significant coefficients.

    Parameters
    ----------
    feature : pd.Series
        The feature to evaluate.
    target : pd.Series
        The target to evaluate against.
    n_splits : int, optional
        The number of splits to use for k-fold cross validation, by default 5.
    alpha : float, optional
        The alpha value to use for determining significance, by default 0.05.

    Returns
    -------
    avg_mse : float
        The average mean squared error of the feature.
    avg_coef : float
        The average coefficient of the feature.
    significant_count : int
        The number of significant coefficients.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    coefficients = []
    significant_count = 0

    # Determine if feature is numerical or categorical
    is_numerical = np.issubdtype(feature.dtype, np.number)

    # Define preprocessor based on feature type
    if is_numerical:
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), [0])
        ])
    else:
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(), [0])
        ])

    # Create pipeline with preprocessor and logistic regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(penalty=None, solver='saga'))
    ])

    for train_index, test_index in kf.split(feature):
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Reshape for sklearn
        X_train, X_test = X_train.values.reshape(-1, 1), X_test.values.reshape(-1, 1)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Predict and calculate MSE
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)

        # Coefficient from logistic regression
        coef = pipeline.named_steps['classifier'].coef_[0][0]
        coefficients.append(coef)

        # Check significance using statsmodels (requires handling of categorical
        # feature differently)
        X_train_sm = sm.add_constant(X_train) if \
                        is_numerical else \
                        sm.add_constant(OneHotEncoder().fit_transform(X_train))
        model_sm = sm.Logit(y_train, X_train_sm)
        result = model_sm.fit(disp=False)
        try:
            p_value = result.pvalues[1 if is_numerical else result.pvalues[1:].min()]

            if p_value < alpha: # Assuming 95% confidence level
                significant_count += 1
        except:
            pass

    avg_mse = np.mean(mse_list)
    avg_coef = np.mean(coefficients)

    return dict(ave_mse_logistic_model=np.round(avg_mse, 5),
                ave_coef_logistic_model=np.round(avg_coef, 5),
                significant_pct=np.round(significant_count/n_splits, 5))

def evaluate_features(df: pd.DataFrame,
                      target_column: str,
                      n_splits: int = 5,
                      alpha: float = 0.05):
    """
    Evaluates all features in a DataFrame by fitting a logistic regression
    model to each feature and the target. The evaluation is done using k-fold
    cross validation. The evaluation metrics are the average mean squared
    error, average coefficient of the feature, and the number of significant
    coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the features and target.
    target_column : str
        The name of the target column.
    n_splits : int, optional
        The number of splits to use for k-fold cross validation, by default 5.
    alpha : float, optional
        The alpha value to use for determining significance, by default 0.05.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame containing the evaluation metrics for each feature.
    """
    # Extract target
    target = df[target_column]
    
    # Initialize a list to hold the results
    results = []

    # suppress convergence warnings


    # Loop through the columns excluding the target
    for feature_name in df.columns:
        try:
            if feature_name != target_column:
                feature = df[feature_name]
                result = evaluate_feature(feature, target, n_splits, alpha)
                result['feature_name'] = feature_name
                results.append(result)
        except Exception as e:
            print(f"Error evaluating feature: {feature_name} - {e}")
            continue

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    results_df.set_index('feature_name', inplace=True)

    return results_df
