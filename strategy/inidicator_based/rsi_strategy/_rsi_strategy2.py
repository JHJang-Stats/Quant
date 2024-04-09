import numpy as np
import pandas as pd
from ...base_strategy import Strategy

class RSIStrategy2(Strategy):
    """
    Implements a modified Relative Strength Index (RSI) strategy based on configurable RSI ranges.

    The strategy generates:
    - A long signal (1) when the RSI is within a specified upper range, suggesting bullish momentum.
    - A short signal (-1) when the RSI is within a specified lower range, indicating bearish momentum.
    - A flat signal (0) otherwise, indicating no clear trading signal based on RSI.
    """
    def __init__(self, data, period=14, short_range=(30, 50), long_range=(50, 70)):
        super().__init__(data)
        self.period = period
        self.short_range = short_range
        self.long_range = long_range

    def generate_signals(self):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        self.data['rsi'] = rsi

        self.signals.loc[(self.data['rsi'] > self.long_range[0]) & (self.data['rsi'] <= self.long_range[1]), 'signal'] = 1.0
        # Set short data based on short_range
        self.signals.loc[(self.data['rsi'] >= self.short_range[0]) & (self.data['rsi'] <= self.short_range[1]), 'signal'] = -1.0

        self.validate_signals()