import pandas as pd
import numpy as np
import warnings

class ColDType:
    def __init__(self, s:pd.Series,
                 categorical_cutoff: float = 0.15,
                 verbose:bool=False):
        self.s = s
        self.verbose = verbose
        self.categorical_cutoff = categorical_cutoff
        
        self.is_binary = None
        self.is_categorical = None
        self.is_date = None
        self.is_finite_numeric = None
        self.is_other_numeric = None
        self.is_object = None

        self.is_type_known = False
        self.sb_dype = None

    def __post_init__(self):
        # set self.is_empty
        if len(self.s) == 0:
            self.is_empty = True
        else:
            self.is_empty = False

        # set self.distinct
        self.distinct = self.s.drop_duplicates().reset_index(drop=True)

    def set_type(self,
                 dtype: str = None):
        """
        This function sets the type of the series. If the dtype is not
        specified, then the function will try to determine the type of the
        series. If the dtype is specified, then the function will set the
        type of the series to the specified dtype.
        """
        assert not self.is_empty, "Series is empty, so you cannot set the type."
        if dtype is None:
            pass
        elif dtype == 'binary':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                True, False, False, False, False, False
            self.is_type_known = True
            self.sb_dtype = 'binary'
        elif dtype == 'categorical':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, True, False, False, False, False
            self.is_type_known = True
            self.sb_dtype = 'categorical'
        elif dtype == 'date':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, True, False, False, False
            self.is_type_known = True
            self.sb_dtype = 'date'
        elif dtype == 'finite_numeric':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, True, False, False
            self.is_type_known = True
            self.sb_dtype = 'finite_numeric'
        elif dtype == 'other_numeric':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, False, True, False
            self.is_type_known = True
            self.sb_dtype = 'other_numeric'
        elif dtype == 'object':
            self.is_binary, self.is_categorical, self.is_date, \
            self.is_finite_numeric, self.is_other_numeric, self.is_object = \
                False, False, False, False, False, True
            self.is_type_known = True
            self.sb_dtype = 'object'
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
                self.s =  self.s.fillna(char_nan)
            elif str_mask.mean() > string_threshold:
                self.s =  self.s.astype(str).replace(char_nan.lower(), char_nan)
        
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

    def _is_binary(self):
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
            exit()

        if self.verbose:
            print(f"Checking if {self.s.name} is binary...")

        # handle NaN values
        self.s = self._handle_nan()

        # if the series is empty, it is not binary
        if self.is_empty:
            self.is_binary = False
            
        # get the unique values
        unique_values = self.distinct

        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.is_binary = False
        
        # if there are two unique values, and those values are 0 and 1,
        # True and False, "Yes" and "No", "Y" and "N", "T" and "F", or 
        # "TRUE" and "FALSE", then the series is binary
        # (any of the strings can be upper or lower case in any combination)
        if len(unique_values) == 2:
            if self.verbose:
                print(f"Column {self.s.name} has two unique values.")

            # if both values are strings, make them lowercased
            if isinstance(unique_values[0], str) and isinstance(unique_values[1], str):
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
                print(f"Column {self.s.name} has only one unique value: \
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
                print(f"Column {self.s.name} has more than two unique values, \
so it is not binary.")
            self.is_binary = False
    
    def _is_finite_numeric(self):
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
            exit()

        if self.verbose:
            print(f"Checking if {self.s.name} is finite numeric...")

        assert isinstance(self.s, pd.Series), "s must be a pandas series"

        # handle NaN values
        if self.verbose:
            print("Handling NaN values...")
        s = self._handle_nan()
        
        # get the unique values
        unique_values = s.unique

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
        elif isinstance(s.dtype, pd.DatetimeTZDtype) or isinstance(
            s[0], pd.Timestamp) or np.issubdtype(s.dtype, np.datetime64):
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
    def _is_date(self):
        """
        Determines whether a series is a date or not. If the series is a date,
        then it is a date. Otherwise, it is not a date.
        """
        if self.is_type_known:
            exit()

        if self.verbose:
            print(f"Checking if {self.s.name} is date...")

        # handle NaN values
        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.set_type('date')
        
        unique_values = self.distinct

        # make sure the series is not binary
        if self.is_binary:
            self.is_date = False

        # make sure the series is not finite numeric
        elif self.is_finite_numeric:
            self.is_date = False

        # if series is numeric, perform further checks
        elif pd.api.types.is_numeric_dtype(self.s):
            try:
                if (self.s < 0).any():
                    self.is_date = False
                elif (self.s > 1e10).any():
                    self.is_date = False
                elif (self.s.dropna() != self.s.dropna().astype(int)).any():
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
                    parsed = pd.to_datetime(self.s, errors='coerce')

                # if any of the parsed dates are 1/1/1970, then it is not a date
                if (parsed.dt.year.eq(1970).any() & 
                    parsed.dt.month.eq(1).any() & 
                    parsed.dt.day.eq(1).any()).any():
                    self.is_date = False

                if pd.isnull(parsed).sum() / self.s.shape[0] > 0.05:
                    self.is_date = False
                # check if parsed results have day, month, year components
                has_components = (~parsed.dt.day.isnull() & \
                                  ~parsed.dt.month.isnull() & \
                                  ~parsed.dt.year.isnull()).all()

                if has_components:
                    self.set_type('date')
                # make sure the series is not categorical
                elif len(unique_values) < max(self.categorical_cutoff * \
                                              self.s.shape[0], 1000):
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
    def _is_categorical(self):
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
            exit()

        if self.verbose:
            print(f"Checking if {self.s.name} is categorical...")

        # test if the column name indicates that it is a date column
        if self._is_date_col_name():
            self.is_categorical = False
        
        # get the unique values
        unique_values = self.distinct

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
    def _is_other_numeric(self):
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
            exit()

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
        if self.is_binary():
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
    def _is_object(self):
        """
        Determines whether a series is an object or not. If the series is not:
            - binary
            - date
            - categorical
            - numeric (finite or other)
        then it is an object. An object is a catch-all for any series that is
        not binary, date, categorical, or numeric.
        """
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