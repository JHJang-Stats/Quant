import numpy as np
import pandas as pd
from ...base_strategy import Strategy


class ADLStrategy(Strategy):
    def __init__(self, data):
        super().__init__(data)
    
    def generate_signals(self):
        copy_data = self.data.copy()
        mfm = ((copy_data['close'] - copy_data['low']) - (copy_data['high'] - copy_data['close'])) / (copy_data['high'] - copy_data['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * copy_data['volume']

        copy_data['adl'] = mfv.cumsum()

        self.signals = pd.DataFrame(index=copy_data.index)
        self.signals['signal'] = 0
        self.signals['signal'][1:] = np.where(copy_data['adl'][1:] > copy_data['adl'][:-1].values, 1, -1)
    
        self.validate_signals()
        del copy_data
