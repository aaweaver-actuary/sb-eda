import pandas as pd
import numpy as np
import datetime


def binary_examples():
    return ([
        np.array([True, False, True]),
        np.array(["t", "f", "t"]),
        np.array([1, 0, 1]),
        np.array([1, 1, 1]),
        np.array([0, 0, 0]),
        np.array([0,1]),
        np.array([True]),
        np.array([False]),
        np.array([True, False]),
        np.array([True, True]),
        np.array([False, False]),
        np.array([True, False, True]),
        np.array([False, True, False]),
        np.array(["Yes", "No"]),
        np.array(["y", "n"]),
        np.array(["Y", "N"]),
        np.array(["T", "F"]),
        np.array(["TRUE", "FALSE"]),
        np.array(["true", "false"]),
        np.array(['y']),
        np.array(['n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']),
        np.array(['y', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y']),
        np.array(['t', 't', 't', 't', 't', 't', 't', 't', 't', 't']),
        np.array(['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f']),
        np.array(['yes', 'yes', 'yes', 'yes', 'yes', \
                  'yes', 'yes', 'yes', 'yes', 'yes']),
        np.array(['no', 'no', 'no', 'no', 'no', \
                  'no', 'no', 'no', 'no', 'no']),
        np.array(['true', 'true', 'true', 'true', 'true', \
                  'true', 'true', 'true', 'true', 'true']),
        np.array(['false', 'false', 'false', 'false', 'false', \
                  'false', 'false', 'false', 'false', 'false']),
        np.array(['Yes', 'No', 'Yes', 'No', 'Yes', \
                  'No', 'Yes', 'No', 'Yes', 'No']),
    ])

def date_examples():
    date_examples = [
        np.array(["12/21/2020", "12/22/2020", "12/23/2020"]),
        np.array([datetime.date(2020, 12, 21),
                  datetime.date(2020, 12, 22),
                  datetime.date(2020, 12, 23)]),
        np.array(["2020-01-01"]),
        np.array(["2020-01-01", "2020-01-02"]),
        np.array(["1/1/2020"]),
        np.array(["01JAN2020"]),
    ]
    return date_examples

def categorical_examples():
    categorical_examples = [
        np.array(['a', 'b', 'a']),
        np.array([1, 2, 1]),
        np.random.choice(['a', 'b', 'c'], size=1000),
        np.array(["a", "b", "c"]),
        np.array(["a", "b", "c", "a"]),
        np.random.choice(["a", "b", "c"], size=1000, replace=True),
        np.random.randint(0, 3, 100000),
        np.random.choice([0, 1, 2], size=1000),
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1, 2, 3.0]),
    ] 
    return categorical_examples

def finite_numeric_examples():
    finite_numeric_examples = [
        np.random.randint(0, 100000, size=100000),
        np.arange(100000),
        np.random.randint(0, 100000, size=14),
        np.random.randint(0, 100000, size=100),
        np.random.randint(0, 100000, size=101),
        np.multiply(np.random.randint(0, 100000, size=100000),1.0),
        
    ]
    return finite_numeric_examples

def other_numeric_examples():
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
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(np.random.choice(list(alphabet), size=length)) for i in range(n)]
    
def object_examples():  
    object_examples = [
    char_gen(3, 500),
        char_gen(5, 500),
        char_gen(10, 500),
    ] 
    return object_examples