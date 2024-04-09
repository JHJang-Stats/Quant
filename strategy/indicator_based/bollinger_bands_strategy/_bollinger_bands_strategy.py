import numpy as np
import pandas as pd
from ...base_strategy import Strategy


class BollingerBandsStrategy(Strategy):
    def __init__(self, data, period=20, std_multiplier=2):
        super().__init__(data)
        self.period = period
        self.std_multiplier = std_multiplier

    def generate_signals(self):
        copy_data = self.data.copy()
        copy_data["middle_band"] = copy_data["close"].rolling(window=self.period).mean()
        std_dev = copy_data["close"].rolling(window=self.period).std()

        copy_data["upper_band"] = copy_data["middle_band"] + (
            self.std_multiplier * std_dev
        )
        copy_data["lower_band"] = copy_data["middle_band"] - (
            self.std_multiplier * std_dev
        )

        self.signals["signal"][copy_data["close"] < copy_data["lower_band"]] = 1
        self.signals["signal"][copy_data["close"] > copy_data["upper_band"]] = -1

        self.validate_signals()

        del copy_data
