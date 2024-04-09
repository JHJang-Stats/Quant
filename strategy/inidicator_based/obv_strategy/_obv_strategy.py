import numpy as np
import pandas as pd
from ...base_strategy import Strategy

class OBVStrategy(Strategy):
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        copy_data = self.data.copy()
        copy_data['daily_ret'] = copy_data['close'].diff()
        copy_data['obv'] = 0
        copy_data.loc[copy_data['daily_ret'] > 0, 'obv'] = copy_data['volume']
        copy_data.loc[copy_data['daily_ret'] < 0, 'obv'] = -copy_data['volume']
        copy_data['obv'] = copy_data['obv'].cumsum()

        self.signals['signal'][1:] = np.where(copy_data['obv'][1:] > copy_data['obv'][:-1].values, 1, -1)

        self.validate_signals()

        del copy_data