import fredapi
import pandas as pd
from _api_keys import get_fred_api_key

Fred = fredapi.Fred(api_key=get_fred_api_key())

def get_fred_data(series_id: str,
                    start_date: str ='2016-01-01',
                    end_date: str = None,
                    frequency: str = 'd') -> pd.DataFrame:
        """
        Get data from FRED and return as a pandas DataFrame.
    
        Parameters
        ----------
        series_id : str
            The series id of the data to get.
        start_date : str
            The start date of the data to get, by default '2016-01-01'.
        end_date : str, optional
            The end date of the data to get, by default None. If None, the
            current date is used.
        frequency : str, optional
            The frequency of the data to get, by default 'd'.
    
        Returns
        -------
        pd.DataFrame
            The data from FRED as a pandas DataFrame.
        """
        if end_date is None:
            end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        return Fred.get_series(series_id,
                             observation_start=start_date,
                             observation_end=end_date,
                             frequency=frequency)