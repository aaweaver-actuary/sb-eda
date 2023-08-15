import pandas as pd
import numpy as np
import datetime
from datetime import date


def binary_examples_re():
    return ([
        np.array([1, 0, 1]),
        np.array([1, 0, 1]),
        np.array([1, 0, 1]),
        np.array([1, 1, 1]),
        np.array([0, 0, 0]),
        np.array([0,1]),
        np.array([1]),
        np.array([0]),
        np.array([1, 0]),
        np.array([1, 1]),
        np.array([0, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1]),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
    ])

def date_examples_re():
    date_examples = [
        np.array([date(2020, 12, 21),date(2020, 12, 22),date(2020, 12, 23)]),
        np.array([date(2020, 12, 21),date(2020, 12, 22),date(2020, 12, 23)]),
        np.array([date(2020, 1, 1)]),
        np.array([date(2020, 1, 1), date(2020, 1, 2)]),
        np.array([date(2020, 1, 1)]),
        np.array([date(2020, 1, 1)]),
    ]
    return date_examples

def categorical_examples_re():
    categorical_examples = [
        np.array(['a', 'b', 'a']).astype(pd.Categorical),
        np.array(['1', '2', '1']).astype(pd.Categorical),
        np.random.choice(['a', 'b', 'c'], size=1000).astype(pd.Categorical),
        np.array(["a", "b", "c"]).astype(pd.Categorical),
        np.array(["a", "b", "c", "a"]).astype(pd.Categorical),
        np.random.choice(["a", "b", "c"], size=1000, replace=1).astype(pd.Categorical),
        'abc',
        # skip this one 
        # np.random.randint(0, 3, 100000).astype(str).astype(pd.Categorical),
        np.random.choice([0, 1, 2], size=1000).astype(str).astype(pd.Categorical),
        np.array(['1', '2', '3']).astype(pd.Categorical),
        np.array(['1', '2', '3']).astype(pd.Categorical),
        np.array(['1', '2', '3']).astype(pd.Categorical),
    ] 
    return categorical_examples

def finite_numeric_examples_re():
    finite_numeric_examples = [
        np.random.randint(0, 100000, size=100000),
        np.arange(100000),
        np.random.randint(0, 100000, size=14),
        np.random.randint(0, 100000, size=100),
        np.random.randint(0, 100000, size=101),
        np.multiply(np.random.randint(0, 100000, size=100000),1.0),
        
    ]
    return finite_numeric_examples

def other_numeric_examples_re():
    other_numeric_examples = [
        np.random.randn(100000),
        np.array([1, 2, 3, np.nan]),
        np.array([1.0, 2.0, 3.5]),
        np.array([1, 2, 3.5]),
        np.array([1.0, 2.0, 3.0, np.nan]),
        np.array([1, 2, 3.0, np.nan]),
    ]
    return other_numeric_examples
    
# arbitrarily many character generator:
def char_gen(length:int, n: int):
    """
    Returns a list of n strings of length l, where each string is composed of
    l random characters from the alphabet.
    """
    alphabet = "abcdefghijklm0pqrstuvwxyz"
    return ["".join(np.random.choice(list(alphabet), size=length)) for i in range(n)]
    
def object_examples_re():  
    object_examples = [
    char_gen(3, 500),
        char_gen(5, 500),
        char_gen(10, 500),
    ] 
    return object_examples

def get_examples_re():
    binary = pd.DataFrame({"examples":binary_examples_re()})
    binary["type"] = "binary"

    date = pd.DataFrame({"examples":date_examples_re()})
    date["type"] = "date"

    categorical = pd.DataFrame({"examples":categorical_examples_re()})
    categorical["type"] = "categorical"

    finite_numeric = pd.DataFrame({"examples":finite_numeric_examples_re()})
    finite_numeric["type"] = "finite_numeric"

    other_numeric = pd.DataFrame({"examples":other_numeric_examples_re()})
    other_numeric["type"] = "other_numeric"

    object_ = pd.DataFrame({"examples":object_examples_re()})
    object_["type"] = "object"

    examples = pd.concat([binary,
                          date,
                          categorical,
                          finite_numeric,
                          other_numeric,
                          object_])
    return examples