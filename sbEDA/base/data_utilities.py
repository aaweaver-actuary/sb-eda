"""
# Description

This file contains some utilities for data processing. It extends the functionality
of the pandas library to do some data processing tasks specific to setting up the data
for Small Business predictive modeling.

# Usage

These are simply utility functions, so they can be imported and used as needed. For
example, if you want to use the is_binary function, you can import it like this:

from SBData.data_utilities import is_binary

# Functions

## `is_binary`
This function takes a pandas series and returns a boolean representing whether the
series is binary or not. A series is binary if it has two unique values, or if it
has one unique value and that value is one of: 0, 1, True, False, or an upper or
lower case version of one of: "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE".

## `is_date`
This function takes a pandas series and returns a boolean representing whether the
series is a date or not. A series is a date if it  can be converted to a date using
the pandas to_datetime function.

## `is_categorical`
This function takes a pandas series and returns a boolean representing whether the
series is categorical or not. 

"""

import pandas as pd
import numpy as np

import warnings

from testingdata import *
examples = get_examples()


# first takes a series and returns a boolean representing whether the
# series is binary or not
def is_binary(s: pd.Series) -> bool:
    """
    Determines whether a series is binary or not. If the series has two
    unique values, then it is binary. If the series only has one unique
    value, and that value is one of:
    0, 1, True, False, or an upper or lower case version of one of:
    "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE",
    then it is binary. Otherwise, it is not binary.
    """

    # get the unique values
    unique_values = s.unique()
    
    # if there are two unique values, and those values are 0 and 1,
    # True and False, "Yes" and "No", "Y" and "N", "T" and "F", or 
    # "TRUE" and "FALSE", then the series is binary
    # (any of the strings can be upper or lower case in any combination)
    if len(unique_values) == 2:
        # if both values are strings, make them lowercased
        if isinstance(unique_values[0], str) and isinstance(unique_values[1], str):
            unique_values = [val.lower() for val in unique_values]

        # check if the values are 0 and 1, True and False, or
        # "yes" and "no", "y" and "n", "t" and "f", or
        # "true" and "false"
        if all(i in [0, 1] for i in unique_values):
            return True
        elif all(i in [True, False] for i in unique_values):
            return True
        elif all(i in ["yes", "no"] for i in unique_values):
            return True
        elif all(i in ["y", "n"] for i in unique_values):
            return True
        elif all(i in ["t", "f"] for i in unique_values):
            return True
        elif all(i in ["true", "false"] for i in unique_values):
            return True
        else:
            return False ## false because the values could be any pair of strings or dates, etc
           
        

    # if there is only one unique value, then the series is binary if
    # that value is 0, 1, True, False, or an upper or lower case version of
    # one of: "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE"
    elif len(unique_values) == 1:
        check_val = unique_values[0]
        if isinstance(check_val, str):
            check_val = check_val.upper()
        return (
           check_val in [0, 1, True, False] or
           check_val in ["YES", "NO", "Y", "N", "T", "F", "TRUE", "FALSE"])

    # otherwise, the series is not binary
    else:
        return False
    
# extend the pandas series class to include the is_binary method
pd.Series.is_binary = is_binary 

# function to test the is_binary method
def test_is_binary(t:str, s: pd.Series, expected: bool):
    """
    Tests the is_binary method on a series, and returns True if the
    result matches the expected value, and False otherwise.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"
    assert s.is_binary() == expected, \
        f"""is_binary method is not correct for the {t} series:
{s}
Expected: {expected}
Actual: {s.is_binary()}"""

example_type = 'binary'
# run the true test cases
for i, s in enumerate(examples
                     .loc[examples['type'].eq(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].eq(example_type), 'type']\
        .tolist()[i]
    test_is_binary(t, s, True)

for i, s in enumerate(examples
                     .loc[examples['type'].ne(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].ne(example_type), 'type']\
        .tolist()[i]
    test_is_binary(t, s, False)

# next, we extend the pandas series class to include the is_finite_numeric
def is_finite_numeric(s: pd.Series) -> bool:
    """
    Determines whether a series is finite numeric or not. If the series has
    only integer numeric values, then it is finite numeric. Otherwise, it
    is not finite numeric. If the series is:
      - binary
      - date
      - categorical
    or has missing values, then it is not finite numeric.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # get the unique values
    unique_values = s.unique()

    # error handling this for when it is used below
    def all_floats_are_ints():
        try:
          return (pd.Series(unique_values) -
              pd.Series(unique_values).astype(int)).abs().sum() == 0
        except TypeError:
            return False
        
    def max_distance_between_floats_and_ints_is_one():
       sorted_unique_values = pd.Series(unique_values).sort_values()
       differences = sorted_unique_values.diff().abs()
       return differences.max() == 1

    # if any of these are strings that cannot be converted to numbers,
    # then the series is not finite numeric
    if np.any(
        [
            isinstance(val, str) and not val.isnumeric()
            for val in unique_values
        ]
    ):
        return False
    
    # if the series is binary, then it is not finite numeric
    elif s.is_binary():
        return False

    # if the series is a date, then it is not finite numeric
    elif isinstance(s.dtype, pd.DatetimeTZDtype) or isinstance(
        s[0], pd.Timestamp) or np.issubdtype(s.dtype, np.datetime64):
        return False

    # if the series has missing values, then it is not finite numeric
    elif pd.Series(unique_values).isna().any():
        return False
    
    # if the series consists of consecutive integers, then it is not
    # finite numeric - this is because it is likely a set of codes
    # that are being used to represent categories
    elif max_distance_between_floats_and_ints_is_one():
        # that is, unless there are more than 100 unique values
        if len(unique_values) <= 100:
            return False  ## in this case it is probably a set of codes 
        else:
            return True

    # if the series is all integers, then it is finite numeric
    elif pd.Series(unique_values).dtype == int:
        return True
    
    # this function is defined above
    elif all_floats_are_ints():
        return True

    # otherwise, the series is not finite numeric
    else:
        return False
    
# extend the pandas series class to include the is_finite_numeric method
pd.Series.is_finite_numeric = is_finite_numeric

# function to test the is_finite_numeric method
def test_is_finite_numeric(t:str, s: pd.Series, expected: bool):
    """
    Tests the is_finite_numeric method on a series, and returns True if the
    result matches the expected value, and False otherwise.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"
    assert s.is_finite_numeric() == expected, \
        f"""is_finite_numeric method is not correct for the {t} series:
{s}
Expected: {expected}
Actual: {s.is_finite_numeric()}"""

example_type = 'finite_numeric'
# run the true test cases
for i, s in enumerate(examples
                     .loc[examples['type'].eq(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].eq(example_type), 'type']\
        .tolist()[i]
    test_is_finite_numeric(t, s, True)

for i, s in enumerate(examples
                     .loc[examples['type'].ne(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].ne(example_type), 'type']\
        .tolist()[i]
    test_is_finite_numeric(t, s, False)

# next, we extend the pandas series class to include the is_date method
def is_date(s: pd.Series,
            categorical_cutoff: int = 100) -> bool:
    """
    Determines whether a series is a date or not. If the series is a date,
    then it is a date. Otherwise, it is not a date.
    """
    unique_values = s.unique()
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # make sure the series is not binary
    if s.is_binary():
        return False

    # make sure the series is not finite numeric
    elif s.is_finite_numeric():
        return False

    # if series is numeric, perform further checks
    elif pd.api.types.is_numeric_dtype(s):
        try:
            if (s < 0).any() or \
               (s > 1e10).any() or \
               (s.dropna() != s.dropna().astype(int)).any():
                return False
            elif len(unique_values) <= categorical_cutoff:
                return False
        except TypeError:
            return False

    # try to convert the series to a date
    else:
        try:
            # catch warnings that occur when converting to date
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(s, errors='coerce')

            # if any of the parsed dates are 1/1/1970, then it is not a date
            if (parsed.dt.year.eq(1970).any() & 
                parsed.dt.month.eq(1).any() & 
                parsed.dt.day.eq(1).any()).any():
                return False

            if pd.isnull(parsed).sum() / s.shape[0] > 0.05:  # more than 10% could not be parsed as dates
                return False
            # check if parsed results have day, month, year components
            has_components = (~parsed.dt.day.isnull() & \
                              ~parsed.dt.month.isnull() & \
                              ~parsed.dt.year.isnull()).all()

            if has_components:
                return True
            # make sure the series is not categorical
            elif len(unique_values) < categorical_cutoff:
                return False
            
            else:
                return False
        except (TypeError,
                ValueError,
                OverflowError,
                AttributeError,
                pd.errors.OutOfBoundsDatetime):
            return False

    
# extend the pandas series class to include the is_date method
pd.Series.is_date = is_date

# # function to test the is_date method
def test_is_date(t:str, s: pd.Series, expected: bool):
    """
    Tests the is_date method on a series, and returns True if the
    result matches the expected value, and False otherwise.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"
    assert s.is_date() == expected, \
        f"""is_date method is not correct for the {t} series:
{s}
Expected: {expected}
Actual: {s.is_date()}"""

example_type = 'date'
# run the true test cases
for i, s in enumerate(examples
                     .loc[examples['type'].eq(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].eq(example_type), 'type']\
        .tolist()[i]
    test_is_date(t, s, True)

for i, s in enumerate(examples
                     .loc[examples['type'].ne(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].ne(example_type), 'type']\
        .tolist()[i]
    test_is_date(t, s, False)


# next, we extend the pandas series class to include the is_categorical
# method
def is_categorical(s: pd.Series,
                   categorical_cutoff: int = 100) -> bool:
    """
    Determines whether a series is categorical or not. If the series has
    less than (s.shape * categorical_cutoff) unique values, then it is
    categorical. If the series is binary, then it is not categorical.
    If the series is a date, then it is not categorical. If the series
    has more than (s.shape * categorical_cutoff) unique values, then it is
    not categorical.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # get the unique values
    unique_values = s.unique()

    # error handling this for when it is used below
    def max_distance_between_floats_and_ints_is_one():
       sorted_unique_values = pd.Series(unique_values).sort_values()
       differences = sorted_unique_values.diff().abs()
       return differences.max() == 1

    # if the series is binary, then it is not categorical
    if s.is_binary():
        # print('binary')
        return False
    
    # if there are any nan values, then the series is not categorical
    elif s.isnull().any():
        return False
    
    # if the series is a float type with a non-zero fraction, then it is
    # not categorical
    elif pd.api.types.is_float_dtype(s) and \
         s.dropna().apply(lambda x: x - int(x)).any():
        return False

    # if the series is a date, then it is not categorical
    elif s.is_date():
        # print('date')
        return False
    
    # if the series is finite numeric, then it is not categorical
    elif s.is_finite_numeric():
        # print('finite numeric')
        return False

    # if the series has less than categorical_cutoff unique
    # values, then it is categorical
    elif unique_values.shape[0] < categorical_cutoff:
        return True
    
    # if there are fewer than 50 unique values, then the series is
    # categorical regardless of the categorical_cutoff
    elif unique_values.shape[0] < 50:
        return True
    
    # the series is a set of integers whose max distance between
    # consecutive integers is 1, then the series is categorical
    elif max_distance_between_floats_and_ints_is_one():
        # print('max distance between floats and ints is one')
        return True

    # otherwise, the series is not categorical
    else:
        return False
    
# extend the pandas series class to include the is_categorical method
pd.Series.is_categorical = is_categorical

# # function to test the is_categorical method
def test_is_categorical(t:str, s: pd.Series, expected: bool):
    """
    Tests the is_categorical method on a series, and returns True if the
    result matches the expected value, and False otherwise.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"
    assert s.is_categorical() == expected, \
        f"""is_categorical method is not correct for the {t} series:
s: {s}
s.unique().size: {s.unique().size}
Expected: {expected}
Actual: {s.is_categorical()}"""

example_type = 'categorical'
# run the true test cases
for i, s in enumerate(examples
                     .loc[examples['type'].eq(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].eq(example_type), 'type']\
        .tolist()[i]
    test_is_categorical(t, s, True)

for i, s in enumerate(examples
                     .loc[examples['type'].ne(example_type), 'examples']
                     .tolist()):
    s = pd.Series(s)
    t = examples\
        .loc[examples['type'].ne(example_type), 'type']\
        .tolist()[i]
    test_is_categorical(t, s, False)

# next, we extend the pandas series class to include the is_other_numeric method, which
# serves as a catch-all for numeric series that are not binary, date, categorical, or
# finite numeric
def is_other_numeric(s: pd.Series) -> bool:
    """
    Determines whether a series is other numeric or not. If the series is
    not:
        - binary
        - date
        - categorical
        - finite numeric
    but the series is numeric, then it is other numeric. Otherwise, it is
    not other numeric.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # if the series is binary, then it is not other numeric
    if s.is_binary():
        return False

    # if the series is a date, then it is not other numeric
    elif s.is_date():
        return False

    # if the series is categorical, then it is not other numeric
    elif s.is_categorical():
        return False

    # if the series is finite numeric, then it is not other numeric
    elif s.is_finite_numeric():
        return False

    # if the series is numeric, then it is other numeric
    elif np.issubdtype(s.dtype, np.number):
        return True

    # otherwise, the series is not other numeric
    else:
        return False
    
# extend the pandas series class to include the is_other_numeric method
pd.Series.is_other_numeric = is_other_numeric

# test the is_other_numeric method
# try:
#     test_methods("other_numeric")
# except:
#     print("is_other_numeric method is not correct")
#     raise

# next, we extend the pandas series class to include the is_object method, which
# serves as a catch-all for any series that is not binary, date, categorical, or
# numeric
def is_object(s:pd.Series) -> bool:
    """
    Determines whether a series is an object or not. If the series is not:
        - binary
        - date
        - categorical
        - numeric (finite or other)
    then it is an object. An object is a catch-all for any series that is
    not binary, date, categorical, or numeric.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # if the series is binary, then it is not an object
    if s.is_binary():
        return False
    elif s.is_date():
        return False
    elif s.is_categorical():
        return False
    elif s.is_finite_numeric():
        return False
    elif s.is_other_numeric():
        return False
    else:
        return True
    
# extend the pandas series class to include the is_object method
pd.Series.is_object = is_object

# test the is_object method
# try:
#     test_methods("object")
# except:
#     print("is_object method is not correct")
#     raise

def sb_dtype(s:pd.Series) -> str:
    """
    Process the series data type into Small Business categories. This function
    determines the series Small Business data type for a given series. The 
    The Small Business data type is one of:
        - binary
        - date
        - categorical
        - finite numeric
        - other numeric
        - object

    If the series data type cannot be determined, then a ValueError is
    raised.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    if s.is_binary():
        return "binary"
    elif s.is_date():
        return "date"
    elif s.is_categorical():
        return "categorical"
    elif s.is_finite_numeric():
        return "finite_numeric"
    elif s.is_other_numeric():
        return "other_numeric"
    elif s.is_object():
        return "object"
    else:
        errormsg = "series cannot be coerced to one of the six data types"
        raise ValueError(errormsg)
    
# extend the pandas series class to include the sb_dtype method
pd.Series.sb_dtype = sb_dtype

# test the sb_dtype method
# try:
#     assert pd.Series([0, 1]).sb_dtype() == 'binary'
#     assert pd.Series(['2019-01-01']).sb_dtype() == 'date'
#     assert pd.Series(['loan_status']).sb_dtype() == 'categorical'
#     assert pd.Series([1, 2, 3]).sb_dtype() == 'finite_numeric'
#     assert pd.Series([1, 2, 3.8]).sb_dtype() == 'other_numeric'
# except AssertionError:
#     print("sb_dtype method is not correct")
#     raise    

##########################################################################################
# In this next section, we will extend the pandas dataframe class to include methods for
# formatting each of the six Small Business data types. These methods will be used to
# format the series of the dataframe.
##########################################################################################

def format_binary(s:pd.Series) -> pd.Series:
    """
    Formats a binary series. If the series is binary, then it is formatted
    as an 8-bit integer. Otherwise, it is ignored.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # if the series is not binary, then end the function
    if not s.is_binary():
        return

    # otherwise, format the series as an 8-bit integer:

    # map possible binary values to 0 and 1
    binary_map = {
        0: 0,
        1: 1,
        True: 1,
        False: 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "t": 1,
        "f": 0,
        "true": 1,
        "false": 0,
    }

    # if the values in the series are strings, then convert them to lower case
    # before running them through the binary map
    if s.dtype == "object" or s.dtype == "string":
        s = s.str.lower()

    # map the values in the series to 0 and 1
    s = s.map(binary_map)

    # convert the series to an 8-bit integer
    s = s.astype("int8")

    # return the series
    return s

# extend the pandas dataframe class to include the format_binary method
pd.DataFrame.format_binary = format_binary

# test the format_binary method
# try:
#     assert pd.Series([0, 1]).format_binary().drop_duplicates().sort_values().tolist() == [0, 1]
#     assert pd.Series([True, False]).format_binary().drop_duplicates().sort_values().tolist() == [0, 1]
#     assert pd.Series(["yes", "no"]).format_binary().drop_duplicates().sort_values().tolist() == [0, 1]
#     assert pd.Series(["y", "n"]).format_binary().drop_duplicates().sort_values().tolist() == [0, 1]
#     assert pd.Series(["t", "f"]).format_binary().drop_duplicates().sort_values().tolist() == [0, 1]
#     assert pd.Series(["true", "false"]).format_binary().drop_duplicates().sort_values().tolist() == [0, 1]
# except AssertionError:
#     print("format_binary method is not correct")
#     raise

def format_date(s:pd.Series) -> pd.Series:
    """
    Formats a date series. If the series has a date type, then it is formatted
    as a datetime. Otherwise, it is ignored.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # if the series is not a date, then end the function
    if not s.is_date():
        return

    # otherwise, format the series as a datetime
    s = pd.to_datetime(s)

    # return the series
    return s

# extend the pandas dataframe class to include the format_date method
pd.DataFrame.format_date = format_date

# test the format_date method
# try:
#     assert pd.Series(["2019-01-01"]).format_date().dtype == "datetime64[ns]"
# except AssertionError:
#     print("format_date method is not correct")
#     raise

def format_categorical(s:pd.Series) -> pd.Series:
    """
    Formats a categorical series. If the series is categorical, then it is
    formatted as a category. Otherwise, it is ignored.
    """
    assert isinstance(s, pd.Series), "s must be a pandas series"

    # if the series is not categorical, then end the function
    if not s.is_categorical():
        return

    # otherwise, format the series as a category
    s = s.astype("category")

    # return the series
    return s