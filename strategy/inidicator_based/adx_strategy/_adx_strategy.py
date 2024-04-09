import numpy as np
import pandas as pd
import talib
from ...base_strategy import Strategy


class ADXStrategy(Strategy):
    def __init__(self, data, adx_period=14, threshold=25):
        super().__init__(data)
        self.adx_period = adx_period
        self.threshold = threshold

    def generate_signals(self):
        copy_data = self.data.copy()
        # Calculate +DI, -DI, and ADX using TA-Lib
        copy_data['+DI'] = talib.PLUS_DI(copy_data['high'], copy_data['low'], copy_data['close'], timeperiod=self.adx_period)
        copy_data['-DI'] = talib.MINUS_DI(copy_data['high'], copy_data['low'], copy_data['close'], timeperiod=self.adx_period)
        copy_data['ADX'] = talib.ADX(copy_data['high'], copy_data['low'], copy_data['close'], timeperiod=self.adx_period)
        # Buy signal: ADX > threshold and +DI > -DI
        self.signals.loc[(copy_data['ADX'] > self.threshold) & (copy_data['+DI'] > copy_data['-DI']), 'signal'] = 1
        # Sell signal: ADX > threshold and +DI < -DI
        self.signals.loc[(copy_data['ADX'] > self.threshold) & (copy_data['+DI'] < copy_data['-DI']), 'signal'] = -1

        self.validate_signals()
        del copy_data