import pandas as pd
import numpy as np
import warnings

class ColDType:
    def __init__(self,
                 s:pd.Series,
                 categorical_cutoff: float = 0.15,
                 verbose:bool=False,
                 binary_na_fill = -9999,
                 categorical_na_fill = 'missing',
                 date_na_fill = pd.Timestamp('2345-12-31',
                                             unit='D',
                                             tz='UTC'),
                 finite_numeric_na_fill = -9999,
                 other_numeric_na_fill = -9999,
                 object_na_fill = 'missing'
                 ):
        self.s = s
        self.verbose = verbose
        self.categorical_cutoff = categorical_cutoff

        self.binary_na_fill = binary_na_fill
        self.categorical_na_fill = categorical_na_fill
        self.date_na_fill = date_na_fill
        self.finite_numeric_na_fill = finite_numeric_na_fill
        self.other_numeric_na_fill = other_numeric_na_fill
        self.object_na_fill = object_na_fill

        self.s_fmt = None # formatted series
        
        self.is_binary = None
        self.is_categorical = None
        self.is_date = None
        self.is_finite_numeric = None
        self.is_other_numeric = None
        self.is_object = None

        self.is_type_known = False
        self.dtype = None
        self.is_empty = None

    def __post_init__(self):
        """
        This function is called after the class is initialized. It sets the
        following attributes:
            - self.is_empty
        """
        # set self.is_empty
        if self.s.shape[0] == 0:
            self.is_empty = True
        else:
            self.is_empty = False

        # handle NaN values
        self._handle_nan()

    def __repr__(self):
        return f"ColDType({self.s.name})"

    def GetS(self,
             replace_na = None,
             drop_na: bool = False) -> pd.Series:
        """
        This function returns the original series. Optionally, you can drop the
        NaN values or replace the NaN values with a specified value.

        If both `drop_na` and `replace_na` are specified, then the function will
        give precedence to `replace_na`.

        Parameters
        ----------
        replace_na : any, optional
            The value to replace the NaN values with. The default is None.
        drop_na : bool, optional
            Whether or not to drop the NaN values. The default is False. If
            `replace_na` is specified, then this parameter is ignored.

        Returns
        -------
        pd.Series
            The original series, optionally with the NaN values replaced or
            dropped.

        Example Usage
        -------------
        >>> s = pd.Series([1, 2, 3, np.nan, 5])
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        4    5.0
        dtype: float64

        >>> cdt = ColDType(s)
        >>> cdt.GetS()
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        4    5.0
        dtype: float64

        >>> cdt.GetS(drop_na=True)
        0    1.0
        1    2.0
        2    3.0
        4    5.0
        dtype: float64

        >>> cdt.GetS(replace_na=0)
        0    1.0
        1    2.0
        2    3.0
        3    0.0
        4    5.0

        >>> cdt.GetS(replace_na=0, drop_na=True)
        0    1.0
        1    2.0
        2    3.0
        3    0.0
        4    5.0
        dtype: float64
        """
        assert not self.is_empty, "Series is empty, so you cannot get the series."

        if replace_na is not None:
            return self.s.fillna(replace_na)
        elif drop_na:
            return self.s.dropna()
        else:
            return self.s

    def set_type(self,
                 dtype: str = None,
                 return_:bool = False) -> str:
        """
        This function sets the type of the series. If the dtype is not
        specified, then the function will try to determine the type of the
        series. If the dtype is specified, then the function will set the
        type of the series to the specified dtype.

        Parameters
        ----------
        dtype : str, optional
            The type to set the series to. The default is None. If the dtype is
            not specified, then the function will try to determine the type of
            the series. 
            
            If the dtype is specified, then the function will set
            the type of the series to the specified dtype.
        return_ : bool, optional
            Whether or not to return the type of the series. The default is
            False. 
            
            Note the use of the underscore in the parameter name.

        Returns
        -------
        str | None
            The type of the series. This is only returned if `return_` is True.

        Example Usage
        -------------
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> cdt = ColDType(s)
        >>> cdt.is_type_known
        False

        >>> cdt.set_type()
        >>> cdt.is_type_known
        True

        >>> cdt.dtype
        'finite_numeric'

        >>> cdt.set_type('categorical')
        >>> cdt.is_type_known
        True

        >>> cdt.dtype
        'categorical'

        >>> cdt.set_type(return_=True)
        'categorical'

        >>> cdt.set_type('andy')
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "<stdin>", line 30, in set_type
        AssertionError: Invalid dtype.
        
        dtype must be one of the following: 
        'binary', 'categorical', 'date', 'finite_numeric', \
        'other_numeric', 'object', None.
        
        You specified dtype = 'andy'.
        """
        assert not self.is_empty, "Series is empty, so you cannot set the type."
        assert dtype in ['binary', 'categorical', 'date', 'finite_numeric',
                         'other_numeric', 'object', None], \
f"""Invalid dtype.

dtype must be one of the following:
'binary', 'categorical', 'date', \
'finite_numeric', 'other_numeric', 'object', None.

You specified dtype = '{dtype}'."""
        if dtype is None:
            pass
        elif dtype == 'binary':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                True, False, False, False, False, False
            self.is_type_known = True
            self.dtype = 'binary'
        elif dtype == 'categorical':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, True, False, False, False, False
            self.is_type_known = True
            self.dtype = 'categorical'
        elif dtype == 'date':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, True, False, False, False
            self.is_type_known = True
            self.dtype = 'date'
        elif dtype == 'finite_numeric':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, True, False, False
            self.is_type_known = True
            self.dtype = 'finite_numeric'
        elif dtype == 'other_numeric':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, False, True, False
            self.is_type_known = True
            self.dtype = 'other_numeric'
        elif dtype == 'object':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, False, False, True
            self.is_type_known = True
            self.dtype = 'object'
        else:
            raise ValueError(f"Unknown dtype {dtype}. Please choose from \
'binary', 'categorical', 'date', 'finite_numeric', 'other_numeric', or \
'object'.")        
        
    def get_unique_values(self,
                          replace_na = None,
                          drop_na:bool = False) -> pd.Series:
        """
        This function returns the unique values in the series.
        
        If the series is empty, then the function will raise an error.

        If both `replace_na` and `drop_na` are None, then the function will
        return the unique values in the series, including NaN values.

        Parameters
        ----------
        replace_na : str, optional
            The value to replace NaN values with. The default is None. If
            `replace_na` is not None, then the function will replace NaN
            values with the specified value.
        drop_na : bool, optional
            Whether or not to drop NaN values. The default is False. If
            `drop_na` is True, then the function will drop NaN values.
            If both `replace_na` and `drop_na` are provided, then the
            function will ignore this parameter.

        Returns
        -------
        pd.Series
            The unique values in the series.

        Example Usage
        -------------
        >>> s = pd.Series([1, 1, 3, 4, np.nan, 4])
        >>> cdt = ColDType(s)
        >>> cdt.get_unique_values()
        0    1.0
        1    3.0
        2    np.nan
        3    4.0
        dtype: float64

        >>> cdt.get_unique_values(drop_na=True)
        0    1.0
        1    3.0
        2    4.0
        dtype: float64

        >>> cdt.get_unique_values(replace_na=0)
        0    1.0
        1    3.0
        2    0.0
        3    4.0
        dtype: float64

        >>> cdt.get_unique_values(replace_na=0, drop_na=True)
        0    1.0
        1    3.0
        2    0.0
        3    4.0
        dtype: float64
        """
        assert not self.is_empty, \
            "Series is empty, so you cannot get the unique values."

        # Get the series, replacing or dropping NaN values if necessary
        s = self.GetS(replace_na=replace_na, drop_na=drop_na)

        # Return the unique values
        return s.drop_duplicates().reset_index(drop=True)

    def GetUnique(self,
                  replace_na = None,
                  drop_na:bool = False) -> pd.Series:
        """
        This function is an alias for `get_unique_values`.
        """
        return self.get_unique_values(replace_na=replace_na, drop_na=drop_na)

    # def _handle_nan(self,
    #                 date_parsing_threshold: float = 0.95,
    #                 string_threshold: float = 0.5) -> pd.Series:
    #     """
    #     This function takes a series and returns a series with the NaN values
    #     handled. If the series is character, then the NaN values are replaced
    #     with "NaN". If the series is numeric, then the NaN values are replaced
    #     with the -9999. If the series is a date, then the NaN values are replaced
    #     with 12/31/2345.
    #     """
    #     # Handle NaN values
    #     numeric_nan = self.finite_numeric_na_fill
    #     date_nan = self.date_na_fill
    #     char_nan = self.object_na_fill

    #     warnings.filterwarnings('ignore')
        
    #     if self.is_empty:
    #         return None
        
    #     # Check if all elements can be parsed as dates
    #     parsed_dates = pd.to_datetime(self.GetS(replace_na=pd.Timestamp('12/31/2345'))
    #                                   , errors='coerce')

    #     # Handle date types (replace '.', '', 1 is to allow float)
    #     if parsed_dates.notna().sum() / len(self.s) > date_parsing_threshold:
    #         s =  parsed_dates.fillna(date_nan)
    #     elif self._is_date_col_name(self.s):
    #         self.s =  parsed_dates.fillna(date_nan)
    #     elif self.is_date is not None:
    #         if self.is_date:
    #             self.s =  self.s.fillna(date_nan)
        
    #     # Handle object types
    #     if self.s.dtype == 'object':
    #         str_mask = self.s.map(type).eq(str)
    #         if str_mask.all():
    #             self.s =  self.s.fillna(char_nan)
    #         elif str_mask.mean() > string_threshold:
    #             self.s =  self.s.astype(str).replace(char_nan.lower(), char_nan)
        
    #     # Check if numeric (replace '.', '', 1 is to allow float)
    #     if pd.to_numeric(self.s, errors='coerce').notna().all():
    #         self.s =  pd.to_numeric(self.s).fillna(numeric_nan)

    #     # For numeric and date types
    #     if self.s.dtype in ['int64', 'float64']:
    #         self.s =  self.s.fillna(numeric_nan)
    #     elif np.issubdtype(self.s.dtype, np.datetime64):
    #         self.s = self.s.fillna(date_nan)

    def _is_date_col_name(self) -> bool:
        # check that the name of the column doesn't have some string that
        # indicates it is a date column
        date_names = ['date', 'time', 'dt', 'dat']
        non_date_names = ['year', 'month', 'day', 'yr', 'mo', 'dy', 'update']
        try:
            self.s.name = self.s.name.lower()
        except AttributeError:
            return
        if any([i in self.s.name.lower() for i in date_names]):
            if self.verbose:
                print(f"Column name {self.s.name} is one of {date_names}, and this \
indicates it is a date column.")
            if any([i in self.s.name.lower() for i in non_date_names]):
                if self.verbose:
                    print(f"Column name {self.s.name} is one of {non_date_names}, \
which indicates it is NOT a date column.")
                self.is_date = False
            else:
                if self.verbose:
                    print(f"Column name {self.s.name} is NOT one of {non_date_names}, \
which indicates it is a date column.")
                self.set_type('date')
        else:
            if self.verbose:
                print(f"Column name {self.s.name} is NOT one of {date_names}, which \
indicates it is NOT a date column.")
            self.is_date = False

    def _is_binary(self,
                   s:pd.Series = None,
                   return_:bool = False) -> bool:
        """
        Determines whether a series is binary or not. If the series has two
        unique values, then it is binary. If the series only has one unique
        value, and that value is one of:
        0, 1, True, False, or an upper or lower case version of one of:
        "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE",
        then it is binary. Otherwise, it is not binary.
        """
        # no reason to keep going if the type is already known
        if self.is_type_known:
            return None

        # stop early if self.is_binary is not None
        if self.is_binary is not None:
            return None

        # if a series was passed in, use that instead of self.s, otherwise
        # use self.s
        if s is None:
            s = self.GetS(drop_na=True)
        else:
            s = s.dropna()
        
        # filter out NaN values that have been recoded 
        s = s[~s.eq(self.binary_na_fill) &\
              ~s.eq(self.object_na_fill) &\
              ~s.eq(self.finite_numeric_na_fill) &\
              ~s.eq(self.date_na_fill)]

        if self.verbose:
            print(f"Checking if {s.name} is binary...")

        # if the series is empty, it is not binary
        if self.is_empty:
            self.is_binary = False
            
        # get the unique values
        unique_values = s.dropna()\
                         .drop_duplicates()\
                         .reset_index(drop=True)

        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.is_binary = False
        
        # if there are two unique values, and those values are 0 and 1,
        # True and False, "Yes" and "No", "Y" and "N", "T" and "F", or 
        # "TRUE" and "FALSE", then the series is binary
        # (any of the strings can be upper or lower case in any combination)
        if len(unique_values) == 2:
            if self.verbose:
                print(f"Column {s.name} has two unique values.")

            # if both values are strings, make them lowercased
            if isinstance(unique_values[0], str) and \
               isinstance(unique_values[1], str):
                unique_values = [val.lower() for val in unique_values]

            # check if the values are 0 and 1, True and False, or
            # "yes" and "no", "y" and "n", "t" and "f", or
            # "true" and "false"
            if self.verbose:
                print(f"unique_values: {unique_values}")
            if all(i in [0, 1] for i in unique_values):
                self.set_type('binary')
            elif all(i in [True, False] for i in unique_values):
                self.set_type('binary')
            elif all(i in ["yes", "no"] for i in unique_values):
                self.set_type('binary')
            elif all(i in ["y", "n"] for i in unique_values):
                self.set_type('binary')
            elif all(i in ["t", "f"] for i in unique_values):
                self.set_type('binary')
            elif all(i in ["true", "false"] for i in unique_values):
                self.set_type('binary')
            else:
                # false because the values could be any pair of
                # strings or dates, etc
                self.is_binary = False 
            
        # if there is only one unique value, then the series is binary if
        # that value is 0, 1, True, False, or an upper or lower case version of
        # one of: "Yes", "No", "Y", "N", "T", "F", "TRUE", "FALSE"
        elif len(unique_values) == 1:
            if self.verbose:
                print(f"Column {s.name} has only one unique value: \
{unique_values[0]}")
            check_val = unique_values[0]
            if isinstance(check_val, str):
                check_val = check_val.upper()
            if (check_val in [0, 1, True, False] or \
                check_val in ["YES", "NO", "Y", "N", \
                             "T", "F", "TRUE", "FALSE"]):
                self.set_type('binary')

        # otherwise, the series is not binary
        else:
            if self.verbose:
                print(f"Column {s.name} has more than two unique values, \
so it is not binary.")
            self.is_binary = False
    
    def _is_finite_numeric(self,
                           s:pd.Series = None,
                           return_:bool = False) -> bool:
        """
        Determines whether a series is finite numeric or not. If the series has
        only integer numeric values, then it is finite numeric. Otherwise, it
        is not finite numeric. If the series is:
        - binary
        - date
        - categorical
        or has missing values, then it is not finite numeric.

        If the series has values that all start with 0, should check if the
        values are all integers, and if so, then it is finite numeric.
        """
        if self.is_type_known:
            return None

        # stop early if self.is_finite_numeric is not None
        if self.is_finite_numeric is not None:
            return None

        # if a series was passed in, use that instead of self.s, otherwise
        # use self.s
        if s is None:
            s = self.GetS(drop_na=True)
        else:
            s = s.dropna()
        
        # filter out NaN values that have been recoded 
        s = s[~s.eq(self.binary_na_fill) &\
              ~s.eq(self.object_na_fill) &\
              ~s.eq(self.finite_numeric_na_fill) &\
              ~s.eq(self.date_na_fill)]

        if self.verbose:
            print(f"Checking if {self.s.name} is finite numeric...")

        assert isinstance(self.s, pd.Series), "s must be a pandas series"

        # handle NaN values
        if self.verbose:
            print("Handling NaN values...")
        
        # get the unique values
        unique_values = self.GetUnique(drop_na=True)

        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.is_finite_numeric = False

        # error handling this for when it is used below
        def all_floats_are_ints():
            try:
                s = pd.Series(unique_values)
                # cast to float if not already float
                if s.dtype != float:
                    s = s.astype(float)
                if (s - pd.Series(unique_values).astype(int)).abs().sum() == 0:
                    self.set_type('finite_numeric')
                else:
                    self.is_finite_numeric = False
            except TypeError:
                self.is_finite_numeric = False

        def max_distance_between_floats_and_ints_is_one():
            # try to cast to float
            try:
                s = pd.Series(unique_values).astype(float)
            except TypeError:
                self.is_finite_numeric = False
            sorted_unique_values = s.sort_values()
            differences = sorted_unique_values.diff().abs()
            if differences.max() == 1:
                self.set_type('finite_numeric')
            else:
                self.is_finite_numeric = False

        # if any of these are strings that cannot be converted to numbers,
        # then the series is not finite numeric
        if np.any(
            [
                isinstance(val, str) and not val.isnumeric()
                for val in unique_values
            ]
        ):
            self.is_finite_numeric = False
        
        # if the series is binary, then it is not finite numeric
        elif self.is_binary:
            self.is_finite_numeric = False

        # if the series is a date, then it is not finite numeric
        elif isinstance(self.GetS(drop_na=True).dtype, pd.DatetimeTZDtype) or \
             isinstance(self.GetS(drop_na=True)[0], pd.Timestamp) or \
             np.issubdtype(self.GetS(drop_na=True).dtype, np.datetime64):
            self.is_finite_numeric = False

        # if the series has missing values, then it is not finite numeric
        elif pd.Series(unique_values).isna().any():
            self.is_finite_numeric = False
        
        # if the series consists of consecutive integers, then it is not
        # finite numeric - this is because it is likely a set of codes
        # that are being used to represent categories
        elif max_distance_between_floats_and_ints_is_one():
            # that is, unless there are more than 100 unique values
            if len(unique_values) <= 100:
                ## in this case it is probably a set of codes 
                self.is_finite_numeric = False
            else:
                self.set_type('finite_numeric')

        # if the series is all integers, then it is finite numeric
        elif pd.Series(unique_values).dtype == int:
            self.set_type('finite_numeric')
        
        # this function is defined above
        elif all_floats_are_ints():
            self.set_type('finite_numeric')

        # otherwise, the series is not finite numeric
        else:
            self.is_finite_numeric = False
    


    # next, we extend the pandas series class to include the is_date method
    def _is_date(self,
                 s:pd.Series = None,
                 return_:bool = False) -> bool:
        """
        Determines whether a series is a date or not. If the series is a date,
        then it is a date. Otherwise, it is not a date.
        """
        if self.is_type_known:
            return None

        # stop early if self.is_date is not None
        if self.is_date is not None:
            return None

        # if a series was passed in, use that instead of self.s, otherwise
        # use self.s
        if s is None:
            s = self.GetS(drop_na=True)
        else:
            s = s.dropna()
        
        # filter out NaN values that have been recoded 
        s = s[~s.eq(self.binary_na_fill) &\
              ~s.eq(self.object_na_fill) &\
              ~s.eq(self.finite_numeric_na_fill) &\
              ~s.eq(self.date_na_fill)]

        if self.verbose:
            print(f"Checking if {s.name} is date...")

        # handle NaN values
        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.set_type('date')
        
        unique_values = self.get_unique_values()

        # make sure the series is not binary
        if self.is_binary:
            self.is_date = False

        # make sure the series is not finite numeric
        elif self.is_finite_numeric:
            self.is_date = False

        # if series is numeric, perform further checks
        elif pd.api.types.is_numeric_dtype(s):
            try:
                if (s < 0).any():
                    self.is_date = False
                elif (s > 1e10).any():
                    self.is_date = False
                elif (s != s.astype(int)).any():
                    self.is_date = False
                elif len(unique_values) <= self.categorical_cutoff:
                    self.is_date = False
            except TypeError:
                self.is_date = False

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
                    self.is_date = False

                if pd.isnull(parsed).sum() / s.shape[0] > 0.05:
                    self.is_date = False
                # check if parsed results have day, month, year components
                has_components = (~parsed.dt.day.isnull() & \
                                  ~parsed.dt.month.isnull() & \
                                  ~parsed.dt.year.isnull()).all()

                if has_components:
                    self.set_type('date')
                # make sure the series is not categorical
                elif len(unique_values) < max(self.categorical_cutoff * \
                                              s.shape[0], 1000):
                    self.is_date = False
                
                else:
                    self.is_date = False
            except (TypeError,
                    ValueError,
                    OverflowError,
                    AttributeError,
                    pd.errors.OutOfBoundsDatetime):
                self.is_date = False

    # next, we extend the pandas series class to include the is_categorical
    # method
    def _is_categorical(self,
                        s:pd.Series = None,
                        return_:bool = False) -> bool:
        """
        Determines whether a series is categorical or not. If the series has
        less than `categorical_cutoff` unique values, then it is
        categorical. If the series is binary, then it is not categorical.
        If the series is a date, then it is not categorical. If the series
        is finite numeric, then it is not categorical. Otherwise, it is
        categorical.
        """
        # no reason to keep going if the type is already known
        if self.is_type_known:
            return None

        # stop early if self.is_categorical is not None
        if self.is_categorical is not None:
            return None

        # if a series was passed in, use that instead of self.s, otherwise
        # use self.s
        if s is None:
            s = self.GetS(drop_na=True)
        else:
            s = s.dropna()
        
        # filter out NaN values that have been recoded 
        s = s[~s.eq(self.binary_na_fill) &\
              ~s.eq(self.object_na_fill) &\
              ~s.eq(self.finite_numeric_na_fill) &\
              ~s.eq(self.date_na_fill)]

        if self.verbose:
            print(f"Checking if {self.s.name} is categorical...")

        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.is_categorical = False
        
        # get the unique values
        unique_values = self.get_unique_values()

        # error handling this for when it is used below
        def max_distance_between_floats_and_ints_is_one():
        # return false if `s` is a string or object dtype
            if not pd.api.types.is_numeric_dtype(self.s):
                self.is_categorical = False
            else:
                sorted_unique_values = pd.Series(unique_values).sort_values()
                differences = sorted_unique_values.diff().abs()
                
                if differences.max() == 1:
                    self.set_type('categorical')
                else:
                    self.is_categorical = False

        # if the series is binary, then it is not categorical
        if self.is_binary:
            self.is_categorical = False
        
        # if there are any nan values, then the series is not categorical
        elif self.s.isnull().any():
            self.is_categorical = False
        
        # if the series is a float type with a non-zero fraction, then it is
        # not categorical
        elif pd.api.types.is_float_dtype(self.s) and \
            self.s.dropna().apply(lambda x: x - int(x)).any():
            self.is_categorical = False

        # if the series is a date, then it is not categorical
        elif self.is_date:
            self.is_categorical = False
        
        # if the series is finite numeric, then it is not categorical
        elif self.is_finite_numeric:
            self.is_categorical = False

        # if the series has less than categorical_cutoff unique
        # values, then it is categorical
        elif unique_values.shape[0] < \
            max(self.categorical_cutoff * self.s.shape[0], 1000):
            self.set_type('categorical')
        
        # if there are fewer than 50 unique values, then the series is
        # categorical regardless of the categorical_cutoff
        elif unique_values.shape[0] < 50:
            self.set_type('categorical')
        
        # the series is a set of integers whose max distance between
        # consecutive integers is 1, then the series is categorical
        elif max_distance_between_floats_and_ints_is_one():
            self.set_type('categorical')

        # otherwise, the series is not categorical
        else:
            self.is_categorical = False
    
    # next, we extend the pandas series class to include the is_other_numeric method,
    # which serves as a catch-all for numeric series that are not binary, date,
    # categorical, or finite numeric
    def _is_other_numeric(self,
                          s:pd.Series = None,
                          return_:bool = False) -> bool:
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
        # no reason to keep going if the type is already known
        if self.is_type_known:
            return None

        # stop early if self.is_other_numeric is not None
        if self.is_other_numeric is not None:
            return None

        # if a series was passed in, use that instead of self.s, otherwise
        # use self.s
        if s is None:
            s = self.GetS(drop_na=True)
        else:
            s = s.dropna()
        
        # filter out NaN values that have been recoded 
        s = s[~s.eq(self.binary_na_fill) &\
              ~s.eq(self.object_na_fill) &\
              ~s.eq(self.finite_numeric_na_fill) &\
              ~s.eq(self.date_na_fill)]

        if self.verbose:
            print(f"Checking if {self.s.name} is other numeric...")

        # test if the column name indicates that it is a date column
        if self.verbose:
            print(f"Checking if {self.s.name} is a date column...")
        if self._is_date_col_name():
            self.is_other_numeric = False
        
        # if the series is binary, then it is not other numeric
        if self.verbose:
            print(f"Checking if {self.s.name} is binary...")
        if self.is_binary:
            self.is_other_numeric = False

        # if the series is a date, then it is not other numeric
        elif self.is_date:
            self.is_other_numeric = False

        # if the series is categorical, then it is not other numeric
        elif self.is_categorical:
            self.is_other_numeric = False

        # if the series is finite numeric, then it is not other numeric
        elif self.is_finite_numeric:
            self.is_other_numeric = False

        # if the series is numeric, then it is other numeric
        elif np.issubdtype(self.s.dtype, np.number):
            self.set_type('other_numeric')

        # otherwise, the series is not other numeric
        else:
            self.is_other_numeric = False

    # next, we extend the pandas series class to include the is_object method, which
    # serves as a catch-all for any series that is not binary, date, categorical, or
    # numeric
    def _is_object(self,
                   s:pd.Series = None,
                   return_:bool = False) -> bool:
        """
        Determines whether a series is an object or not. If the series is not:
            - binary
            - date
            - categorical
            - numeric (finite or other)
        then it is an object. An object is a catch-all for any series that is
        not binary, date, categorical, or numeric.
        """
        # no reason to keep going if the type is already known
        if self.is_type_known:
            return None

        # stop early if self.is_object is not None
        if self.is_object is not None:
            return None

        # if a series was passed in, use that instead of self.s, otherwise
        # use self.s
        if s is None:
            s = self.GetS(drop_na=True)
        else:
            s = s.dropna()
        
        # filter out NaN values that have been recoded 
        s = s[~s.eq(self.binary_na_fill) &\
              ~s.eq(self.object_na_fill) &\
              ~s.eq(self.finite_numeric_na_fill) &\
              ~s.eq(self.date_na_fill)]

        if self.verbose:
            print(f"Checking if {self.s.name} is object...")

        assert isinstance(self.s, pd.Series), "s must be a pandas series"

        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.is_object = False
        
        # if the series is binary, then it is not an object
        if self.verbose:
            print(f"Checking if {self.s.name} is any other dtype...")
        if self.is_binary:
            self.is_object = False
        elif self.is_date:
            self.is_object = False
        elif self.is_categorical:
            self.is_object = False
        elif self.is_finite_numeric:
            self.is_object = False
        elif self.is_other_numeric:
            self.is_object = False
        else:
            self.set_type('object')

    def sb_dtype(self, 
                 s:pd.Series = None,
                 return_:bool = False) -> bool:
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
        def bool_return(value=None):
            """
            This function returns the value if return_ is True, otherwise it
            returns None. If no value is passed, then it returns the dtype.
            """
            if return_:
                return value if value is not None else self.dtype
            else:
                return None
        # no reason to keep going if the type is already known
        if self.is_type_known:
            return bool_return()

        # check each data type in order of most specific to least specific
        self._is_binary(s=s)
        if self.is_binary:
            return bool_return('binary')

        self._is_date(s=s)
        if self.is_date:
            return bool_return('date')

        self._is_categorical(s=s)
        if self.is_categorical:
            return bool_return('categorical')

        self._is_finite_numeric(s=s)
        if self.is_finite_numeric:
            return bool_return('finite_numeric')

        self._is_other_numeric(s=s)
        if self.is_other_numeric:
            return bool_return('other_numeric')

        self._is_object(s=s)
        if self.is_object:
            return bool_return('object')

        else:
            errormsg = "series cannot be coerced to one of the six data types"
            raise ValueError(errormsg)

    def format_binary(self):
        """
        Formats a binary series. If the series is binary, then it is formatted
        as an 8-bit integer. Otherwise, it is ignored.
        """
        if not self.is_type_known:
            self.sb_dtype()

        # if the series is not binary, then end the function
        if not self.is_binary:
            return None

        # otherwise, format the series as an 16-bit integer:

        # fill na values
        self.s = self.GetS()

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
        if self.s.dtype == "object" or self.s.dtype == "string":
            self.s = self.s.str.lower()

        # map the values in the series to 0 and 1
        self.s_fmt = self.GetS().map(binary_map).fillna(self.binary_na_fill)

        # convert the series to an 16-bit integer
        self.s_fmt = self.s_fmt.astype("int16")

    def format_date(self):
        """
        Formats a date series. If the series has a date type, then it is formatted
        as a datetime. Otherwise, it is ignored.
        """
        if not self.is_type_known:
            self.sb_dtype()

        # if the series is not date, then end the function
        if not self.is_date:
            return None

        # otherwise, format the series as a datetime
        s = pd.to_datetime(self.GetS(), unit='D')\
              .fillna(self.date_na_fill)
        print(1)
        print(f"1: s.dtype: {s.dtype}\n\ns: {s}\n")
        if s.dtype == "datetime64[ns]":
            self.s_fmt = s
            print(2)
            print(f"2: s_fmt.dtype: {self.s_fmt.dtype}\n\ns_fmt: {self.s_fmt}\n")
        else:
            print(3)
            self.s_fmt = pd.to_datetime(s, unit='D')
            
            print(f"3: s_fmt.dtype: {self.s_fmt.dtype}\n\ns_fmt: {self.s_fmt}\n")

    def format_categorical(self):
        """
        Formats a categorical series. If the series is categorical, then it is
        formatted as a category. Otherwise, it is ignored.
        """
        if not self.is_type_known:
            self.sb_dtype()

        # if the series is not categorical, then end the function
        if not self.is_categorical:
            return None
        
        # if it isn't a string, then convert it to a string
        self.s = self.s.fillna(self.categorical_na_fill)
        if self.s.dtype != "string":
            self.s = self.s.astype("string")

        # otherwise, format the series as a category
        self.s_fmt = self.s.astype("category")

    def format_finite_numeric(self):
        """
        Formats a finite numeric series. If the series is finite numeric, then it
        is formatted as a float. Otherwise, it is ignored.
        """
        if not self.is_type_known:
            self.sb_dtype()

        # if the series is not finite numeric, then end the function
        if not self.is_finite_numeric:
            return None
        
        # otherwise, format the series as a float
        self.s = self.s.fillna(self.finite_numeric_na_fill)
        self.s_fmt = self.s.astype("float").astype(int)

    def format_other_numeric(self):
        """
        Formats an other numeric series. If the series is other numeric, then it
        is formatted as a float. Otherwise, it is ignored.
        """
        if not self.is_type_known:
            self.sb_dtype()

        # if the series is not other numeric, then end the function
        if not self.is_other_numeric:
            return None
        
        # otherwise, format the series as a float
        self.s = self.s.fillna(self.other_numeric_na_fill)
        self.s_fmt = self.s.astype("float")

    def format_object(self):
        """
        Formats an object series. If the series is object, then it is formatted as
        a string. Otherwise, it is ignored.
        """
        if not self.is_type_known:
            self.sb_dtype()

        # if the series is not an object, then end the function
        if not self.is_object:
            return None
        
        # otherwise, format the series as a string
        self.s = self.s.fillna(self.object_na_fill)
        self.s_fmt = self.s.astype("string")

    def format_series(self):
        """
        Formats a series. If the series is categorical, then it is formatted as a
        category. If the series is finite numeric, then it is formatted as a float.
        If the series is other numeric, then it is formatted as a float. If the
        series is object, then it is formatted as a string. Otherwise, it is
        ignored.
        """
        # get the column type
        if not self.is_type_known:
            self.sb_dtype()

        # if the column type is date, then format it as a date
        if self.is_date:
            self.format_date()

        # if the column type is categorical, then format it as a category
        elif self.is_categorical:
            self.format_categorical()
        
        # if the column type is finite numeric, then format it as a float
        elif self.is_finite_numeric:
            self.format_finite_numeric()

        # if the column type is binary, then format it as a binary
        elif self.is_binary:
            self.format_binary()
        
        # if the column type is other numeric, then format it as a float
        elif self.is_other_numeric:
            self.format_other_numeric()
        
        # if the column type is object, then format it as a string
        elif self.is_object:
            self.format_object()
        
        # otherwise, end the function
        else:
            self.s_fmt = self.s.copy().fillna(self.object_na_fill)

###### UPDATE PD.SERIES WITH NEW METHOD ######

# # extend the pandas series class to include the is_binary method
# pd.Series.is_binary = is_binary 

# # extend the pandas series class to include the is_finite_numeric method
# pd.Series.is_finite_numeric = is_finite_numeric

# # extend the pandas series class to include the is_other_numeric method
# pd.Series.is_other_numeric = is_other_numeric

# # extend the pandas series class to include the is_date method
# pd.Series.is_date = is_date

# # extend the pandas series class to include the is_categorical method
# pd.Series.is_categorical = is_categorical

# # extend the pandas series class to include the sb_dtype method
# pd.Series.sb_dtype = sb_dtype

# # extend the pandas series class to include the format_series method
# pd.Series.format_series = format_series

# # extend the pandas series class to include the format_object method
# pd.Series.format_object = format_object

# # extend the pandas series class to include the format_finite_numeric method
# pd.Series.format_finite_numeric = format_finite_numeric

# # extend the pandas series class to include the format_other_numeric method
# pd.Series.format_other_numeric = format_other_numeric

# # extend the pandas series class to include the format_categorical method
# pd.Series.format_categorical = format_categorical

# # extend the pandas series class to include the format_date method
# pd.Series.format_date = format_date


# # extend the pandas series class to include the format_binary method
# pd.Series.format_binary = format_binary