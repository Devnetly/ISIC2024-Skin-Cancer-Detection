import pandas as pd
import os
from typing import Any,Literal
import warnings
from pandas.errors import DtypeWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DtypeWarning)

class History:

    def __init__(self, 
        header : list[str],   
        filename : str,
        eager : bool = False
    ) -> None:
        
        self.filename = filename
        self.header = header + ['split']
        self.eager = eager
        self.df = self._load()

    def _load(self) -> pd.DataFrame:

        df = None

        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.header)
        else:
            df = pd.read_csv(self.filename)

        return df

    def update(self, results : dict[str, Any], split : Literal['train', 'val']) -> None:

        results['split'] = split
        row = pd.DataFrame([results], columns=self.header)
        self.df = pd.concat([self.df, row], ignore_index=True)

        if self.eager:
            self.save()

    def save(self) -> None:
        self.df.to_csv(self.filename, index=False)