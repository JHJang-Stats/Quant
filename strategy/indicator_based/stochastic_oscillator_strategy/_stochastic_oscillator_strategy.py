import numpy as np
import pandas as pd
from ...indicator_based import IndicatorBasedStrategy


class StochasticOscillatorStrategy(IndicatorBasedStrategy):
    def __init__(self, data, period=14, oversold_threshold=20, overbought_threshold=80):
        super().__init__(data)
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signals(self):
        copy_data = self.data.copy()
        low_min = copy_data["low"].rolling(window=self.period).min()
        high_max = copy_data["high"].rolling(window=self.period).max()

        # Calculate the Stochastic Oscillator
        copy_data["%K"] = ((copy_data["close"] - low_min) / (high_max - low_min)) * 100

        self.signals.loc[
            (copy_data["%K"].shift(1) < self.oversold_threshold)
            & (copy_data["%K"] > self.oversold_threshold),
            "signal",
        ] = 1
        self.signals.loc[
            (copy_data["%K"].shift(1) > self.overbought_threshold)
            & (copy_data["%K"] < self.overbought_threshold),
            "signal",
        ] = -1

        self.validate_signals()

        del copy_data
