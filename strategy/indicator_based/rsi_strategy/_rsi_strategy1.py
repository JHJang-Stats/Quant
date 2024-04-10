import numpy as np
import pandas as pd
from ...indicator_based import IndicatorBasedStrategy


class RSIStrategy1(IndicatorBasedStrategy):
    """
    Implements a Relative Strength Index (RSI) strategy.

    The strategy generates buy signals when the RSI falls below a specified threshold
    (indicating oversold conditions) and sell signals when the RSI exceeds another threshold
    (indicating overbought conditions). It is based on momentum and is used to identify
    potential reversals in price.
    """

    def __init__(self, data, period=14, long_threshold=30, short_threshold=70):
        super().__init__(data)
        self.period = period
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signals(self):
        copy_data = self.data.copy()
        delta = copy_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        copy_data["rsi"] = rsi

        self.signals.loc[copy_data["rsi"] < self.long_threshold, "signal"] = 1.0
        self.signals.loc[copy_data["rsi"] > self.short_threshold, "signal"] = -1.0

        self.validate_signals()

        del copy_data
