import numpy as np
import pandas as pd
from ...indicator_based import IndicatorBasedStrategy


class MovingAverageCrossoverStrategy(IndicatorBasedStrategy):
    """
    Implements a simple moving average (SMA) crossover strategy.
    """

    def __init__(self, data, short_window=50, long_window=200):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        copy_data = self.data.copy()
        copy_data["short_mavg"] = (
            copy_data["close"].rolling(window=self.short_window, min_periods=1).mean()
        )
        copy_data["long_mavg"] = (
            copy_data["close"].rolling(window=self.long_window, min_periods=1).mean()
        )

        self.signals["signal"] = np.where(
            copy_data["short_mavg"] > copy_data["long_mavg"], 1.0, 0.0
        )
        self.signals["signal"][: self.long_window] = 0.0

        self.validate_signals()

        del copy_data
