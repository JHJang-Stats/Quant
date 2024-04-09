import numpy as np
import pandas as pd
from ...base_strategy import Strategy


class IchimokuCloudStrategy(Strategy):
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        copy_data = self.data.copy()

        # Ichimoku Cloud Calculations
        high_9 = copy_data["high"].rolling(window=9).max()
        low_9 = copy_data["low"].rolling(window=9).min()
        copy_data["Tenkan-sen"] = (high_9 + low_9) / 2

        high_26 = copy_data["high"].rolling(window=26).max()
        low_26 = copy_data["low"].rolling(window=26).min()
        copy_data["Kijun-sen"] = (high_26 + low_26) / 2

        copy_data["Senkou Span A"] = (
            (copy_data["Tenkan-sen"] + copy_data["Kijun-sen"]) / 2
        ).shift(26)
        high_52 = copy_data["high"].rolling(window=52).max()
        low_52 = copy_data["low"].rolling(window=52).min()
        copy_data["Senkou Span B"] = ((high_52 + low_52) / 2).shift(26)

        # Buy Signal
        self.signals.loc[copy_data["close"] > copy_data["Senkou Span A"], "signal"] = 1
        # Sell/Short Signal
        self.signals.loc[copy_data["close"] < copy_data["Senkou Span B"], "signal"] = -1

        self.validate_signals()

        del copy_data
