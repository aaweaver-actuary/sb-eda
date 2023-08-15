import pandas as pd
import polars as pl

import data_utilities
# from data_utilities import is_binary, \
#                            is_categorical

class base:
    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 exclude_cols: list = None,
                 ):
        self.df = df
        self.df.columns = [x.lower() for x in self.df.columns]
        self.df.columns = [x.replace(' ', '_') for x in self.df.columns]
        self.df.columns = [x.replace('(', '') for x in self.df.columns]
        self.df.columns = [x.replace(')', '') for x in self.df.columns]
        self.df.columns = [x.replace('/', '_') for x in self.df.columns]
        self.df.columns = [x.replace('-', '_') for x in self.df.columns]
        self.df.columns = [x.replace('.', '_') for x in self.df.columns]
        self.df.columns = [x.replace('__', '_') for x in self.df.columns]

        self.target_col = target_col
        self.exclude_cols = exclude_cols

        # classify columns
        self.target = self.df[self.target_col]
        self.features = self.df.drop(columns=[self.target_col] + self.exclude_cols)

        # classify data types in the columns
        self.is_binary = [self.df.drop(columns=[self.target_col] + self.exclude_cols)[col].is_binary() for col in self.features.columns.tolist()]
        self.is_date = [self.df.drop(columns=[self.target_col] + self.exclude_cols)[col].is_date() for col in self.features.columns.tolist()]
        self.is_categorical = [self.df.drop(columns=[self.target_col] + self.exclude_cols)[col].is_categorical() for col in self.features.columns.tolist()]
        self.is_finite_numeric = [self.df.drop(columns=[self.target_col] + self.exclude_cols)[col].is_finite_numeric() for col in self.features.columns.tolist()]
        self.is_other_numeric = [self.df.drop(columns=[self.target_col] + self.exclude_cols)[col].is_other_numeric() for col in self.features.columns.tolist()]
        self.is_object = [self.df.drop(columns=[self.target_col] + self.exclude_cols)[col].is_object() for col in self.features.columns.tolist()]

        # create tables with the data types
        self.binary = [self.df[[col]].to_binary() for col in self.features.loc[:, self.is_binary].columns]



    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    def __str__(self):
        return f'{self.__class__.__name__}'
    
