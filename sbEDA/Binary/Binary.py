from ..base import base
import pandas as pd

class Binary(base):
    def __init__(self, df:pd.DataFrame):
        super().__init__(df)
        
    