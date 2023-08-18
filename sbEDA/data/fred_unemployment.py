import pandas as pd
from fredapi import Fred
from tqdm import tqdm
from _api_keys import get_fred_api_key

Fred = Fred(api_key=get_fred_api_key())

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

def fred_unemployment_data():
    unemployment = pd.read_parquet('./fred_unemployment.parquet')
    state_unemployment_list = []
    state_series = unemployment.loc[unemployment.state_indicator.eq(1), 'id'].drop_duplicates().tolist()
    for s in tqdm(state_series):
        test = get_fred_data(s, frequency='m').reset_index().assign(id=s).set_index(['id']).rename(columns={'index': 'date', 0: 'state_unemployment_rate'})
        state_unemployment_list.append(test)
    state_unemployment = pd.concat(state_unemployment_list)
    state_unemployment.to_parquet('./fred_state_unemployment_pull.parquet')

    county_unemployment_list = []
    county_series = unemployment.loc[unemployment.county_indicator.eq(1), 'id'].drop_duplicates().tolist()
    for s in tqdm(county_series):
        test = get_fred_data(s, frequency='m').reset_index().assign(id=s).set_index(['id']).rename(columns={'index': 'date', 0: 'county_unemployment_rate'})
        county_unemployment_list.append(test)
    county_unemployment = pd.concat(county_unemployment_list)
    county_unemployment.to_parquet('./fred_county_unemployment_pull.parquet')

    msa_unemployment_list = []
    msa_series = unemployment.loc[unemployment.msa_indicator.eq(1), 'id'].drop_duplicates().tolist()
    for s in tqdm(msa_series):
        test = get_fred_data(s, frequency='m').reset_index().assign(id=s).set_index(['id']).rename(columns={'index': 'date', 0: 'msa_unemployment_rate'})
        msa_unemployment_list.append(test)
    msa_unemployment = pd.concat(msa_unemployment_list)
    msa_unemployment.to_parquet('./fred_msa_unemployment_pull.parquet')

    county = pd.read_parquet('./fred_county_unemployment_pull.parquet').rename(columns={'county_unemployment_rate': 'unemployment_rate'})
    msa = pd.read_parquet('./fred_msa_unemployment_pull.parquet').rename(columns={'msa_unemployment_rate': 'unemployment_rate'})
    state = pd.read_parquet('./fred_state_unemployment_pull.parquet').rename(columns={'state_unemployment_rate': 'unemployment_rate'})
    pull = pd.concat([county, msa, state])
    unemployment = pd.read_parquet('./fred_unemployment.parquet')\
                    .reset_index()\
                    .drop(columns='series id')\
                    .set_index('id')\
                    .join(pull, how='left')\
                    .reset_index()

    county_unemployment = unemployment\
                            .loc[unemployment.county_indicator.eq(1), ['location', 'date', 'unemployment_rate']]\
                            .rename(columns={'location': 'county', 'unemployment_rate': 'county_unemployment_rate'})\
                            .reset_index(drop=True)\
                            .assign(state=lambda x: x.county.str.split(',').str[1].str.strip(),
                                    county=lambda x: x.county.str.split(',').str[0].str.strip())\
                            .assign(county=lambda x: x.county.str.replace(' county', ''))
    county_unemployment.to_parquet('./fred_county_unemployment.parquet')

    msa_unemployment = unemployment\
                            .loc[unemployment.msa_indicator.eq(1), ['location', 'date', 'unemployment_rate']]\
                            .rename(columns={'location': 'msa', 'unemployment_rate': 'msa_unemployment_rate'})\
                            .reset_index(drop=True)\
                            .assign(state=lambda x: x.msa.str.split(',').str[1].str.strip(),
                                    msa=lambda x: x.msa.str.split(',').str[0].str.strip())\
                            .assign(msa=lambda x: x.msa.str.replace(' MSA', ''))
    msa_unemployment['state'] = msa_unemployment['state'].str.replace(' (msa)', '')
    msa_unemployment.to_parquet('./fred_msa_unemployment.parquet')

    state_unemployment = unemployment\
                            .loc[unemployment.state_indicator.eq(1), ['location', 'date', 'unemployment_rate']]\
                            .rename(columns={'location': 'state', 'unemployment_rate': 'state_unemployment_rate'})\
                            .reset_index(drop=True)\
                            .assign(state=lambda x: x.state.str.strip())
    state_unemployment.to_parquet('./fred_state_unemployment.parquet')

if __name__ == '__main__':
    fred_unemployment_data()