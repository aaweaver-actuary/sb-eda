import pandas as pd

class base:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.df.columns = [x.lower() for x in self.df.columns]
        self.df.columns = [x.replace(' ', '_') for x in self.df.columns]
        self.df.columns = [x.replace('(', '') for x in self.df.columns]
        self.df.columns = [x.replace(')', '') for x in self.df.columns]
        self.df.columns = [x.replace('/', '_') for x in self.df.columns]
        self.df.columns = [x.replace('-', '_') for x in self.df.columns]
        self.df.columns = [x.replace('.', '_') for x in self.df.columns]
        self.df.columns = [x.replace('__', '_') for x in self.df.columns]
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    def __str__(self):
        return f'{self.__class__.__name__}'
    
