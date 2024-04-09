import numpy as np
import pandas as pd
from ..base_strategy import Strategy

class VanillaStrategy(Strategy):
    def __init__(self, data, start_date=None):
        super().__init__(data)
        self.start_date = pd.to_datetime(start_date)

    def generate_signals(self):
        if self.start_date is not None:
            self.signals.loc[pd.to_datetime(self.signals.index, unit='ms') >= self.start_date, 'signal'] = 1
        else:
            self.signals['signal'] = 1
        self.validate_signals()
