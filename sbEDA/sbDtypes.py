import pandas as pd
import numpy as np
import warnings

class ColDType:
    def __init__(self, s:pd.Series,
                 verbose:bool=False):
        self.s = s
        self.verbose = verbose
        
        self.is_binary = None
        self.is_categorical = None
        self.is_date = None
        self.is_finite_numeric = None
        self.is_other_numeric = None
        self.is_object = None

        self.is_type_known = False

    self.__post_init__(self):
        if len(self.s) == 0:
            self.is_empty = True
        else:
            self.is_empty = False

    self.set_type(self,
                  dtype: str = None):
        """
        This function sets the type of the series. If the dtype is not
        specified, then the function will try to determine the type of the
        series. If the dtype is specified, then the function will set the
        type of the series to the specified dtype.
        """
        assert !s.is_empty, "Series is empty, so you cannot set the type."
        if dtype is None:
            pass
        elif dtype == 'binary':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                True, False, False, False, False, False
            self.is_type_known = True
        elif dtype == 'categorical':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, True, False, False, False, False
            self.is_type_known = True
        elif dtype == 'date':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, True, False, False, False
            self.is_type_known = True
        elif dtype == 'finite_numeric':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, True, False, False
            self.is_type_known = True
        elif dtype == 'other_numeric':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, False, True, False
            self.is_type_known = True
        elif dtype == 'object':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, False, False, True
            self.is_type_known = True
        else:
            raise ValueError(f"Unknown dtype {dtype}. Please choose from \
'binary', 'categorical', 'date', 'finite_numeric', 'other_numeric', or \
'object'.")        
        

    def _handle_nan(self,
                    numeric_nan: float = -9999,
                    date_nan: pd.Timestamp = pd.Timestamp('2999-12-31'),
                    char_nan: str = "NaN",
                    date_parsing_threshold: float = 0.95,
                    string_threshold: float = 0.5) -> pd.Series:
        """
        This function takes a series and returns a series with the NaN values
        handled. If the series is character, then the NaN values are replaced
        with "NaN". If the series is numeric, then the NaN values are replaced
        with the -9999. If the series is a date, then the NaN values are replaced
        with 12/31/2999.
        """
        warnings.filterwarnings('ignore')
        
        if self.is_empty:
            return self.s
        
        # Check if all elements can be parsed as dates
        parsed_dates = pd.to_datetime(self.s, errors='coerce')
        if parsed_dates.notna().sum() / len(self.s) > date_parsing_threshold:
            self.s =  parsed_dates.fillna(date_nan)
        elif self._is_date_col_name(self.s):
            self.s =  parsed_dates.fillna(date_nan)
        elif self.is_date is not None:
            if self.is_date:
                self.s =  self.s.fillna(date_nan)
        
        # Handle object types
        if self.s.dtype == 'object':
            str_mask = self.s.map(type).eq(str)
            if str_mask.all():
                self.s =  self.s.fillna("NaN")
            elif str_mask.mean() > string_threshold:
                self.s =  self.s.astype(str).replace('nan', "NaN")
        
        # Check if numeric (replace '.', '', 1 is to allow float)
        if pd.to_numeric(self.s, errors='coerce').notna().all():
            self.s =  pd.to_numeric(self.s).fillna(numeric_nan)

        # For numeric and date types
        if self.s.dtype in ['int64', 'float64']:
            self.s =  self.s.fillna(numeric_nan)
        elif np.issubdtype(self.s.dtype, np.datetime64):
            self.s = self.s.fillna(date_nan)

    def _is_date_col_name(self) -> bool:
        # check that the name of the column doesn't have some string that
        # indicates it is a date column
        date_names = ['date', 'time', 'dt', 'dat']
        non_date_names = ['year', 'month', 'day', 'yr', 'mo', 'dy', 'update']
        if any([i in self.s.name.lower() for i in date_names]):
            if self.verbose:
                print(f"Column name {self.s.name} is one of {date_names}, and this \
indicates it is a date column.")
            if any([i in self.s.name.lower() for i in non_date_names]):
                if self.verbose:
                    print(f"Column name {self.s.name} is one of {non_date_names}, which \
indicates it is NOT a date column.")
                self.is_date = False
            else:
                if self.verbose:
                    print(f"Column name {self.s.name} is NOT one of {non_date_names}, \
which indicates it is a date column.")
                self.is_date = True
        else:
            if self.verbose:
                print(f"Column name {self.s.name} is NOT one of {date_names}, which \
indicates it is NOT a date column.")
            self.is_date = False

    def _is_binary(self):
        """
        Determines whether a series is binary or not. If the series has two
        unique values, then it is binary. If the series only has one unique
        value, and that value is one of:
        0, 1, True, False, or an upper or lower case version of one of:
        "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE",
        then it is binary. Otherwise, it is not binary.
        """
        if self.verbose:
            print(f"Checking if {self.s.name} is binary...")

        # handle NaN values
        self.s = _handle_nan(self.s)

        # if the series is empty, it is not binary
        if self.is_empty:
            self.is_binary = False
            

        # get the unique values
        unique_values = s.unique()

        # test if the column name indicates that it is a date column
        if _is_date_col_name(s, verbose=verbose):
            return False
        
        # if there are two unique values, and those values are 0 and 1,
        # True and False, "Yes" and "No", "Y" and "N", "T" and "F", or 
        # "TRUE" and "FALSE", then the series is binary
        # (any of the strings can be upper or lower case in any combination)
        if len(unique_values) == 2:
            if verbose:
                print(f"Column {s.name} has two unique values.")

            # if both values are strings, make them lowercased
            if isinstance(unique_values[0], str) and isinstance(unique_values[1], str):
                unique_values = [val.lower() for val in unique_values]

            # check if the values are 0 and 1, True and False, or
            # "yes" and "no", "y" and "n", "t" and "f", or
            # "true" and "false"
            if verbose:
                print(f"unique_values: {unique_values}")
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
                # false because the values could be any pair of
                # strings or dates, etc
                return False 
            
        # if there is only one unique value, then the series is binary if
        # that value is 0, 1, True, False, or an upper or lower case version of
        # one of: "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE"
        elif len(unique_values) == 1:
            if self.verbose:
                print(f"Column {self.s.name} has only one unique value: \
{unique_values[0]}")
            check_val = unique_values[0]
            if isinstance(check_val, str):
                check_val = check_val.upper()
            return (
            check_val in [0, 1, True, False] or
            check_val in ["YES", "NO", "Y", "N", "T", "F", "TRUE", "FALSE"])

        # otherwise, the series is not binary
        else:
            if self.verbose:
                print(f"Column {self.s.name} has more than two unique values, \
so it is not binary.")
            return False
    


###### UPDATE PD.SERIES WITH NEW METHOD ######

# extend the pandas series class to include the is_binary method
pd.Series.is_binary = is_binary 