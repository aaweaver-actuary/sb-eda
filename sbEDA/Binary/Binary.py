from ...base import base
import pandas as pd

class Binary(base):
    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 exclude_cols: list = None):
        super().__init__(df, target_col, exclude_cols)
