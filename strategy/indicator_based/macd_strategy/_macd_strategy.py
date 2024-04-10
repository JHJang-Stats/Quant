import numpy as np
import pandas as pd
from ...indicator_based import IndicatorBasedStrategy


class MACDStrategy(IndicatorBasedStrategy):
    def __init__(
        self, data, short_ema_period=12, long_ema_period=26, signal_line_period=9
    ):
        super().__init__(data)
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.signal_line_period = signal_line_period

    def generate_signals(self):
        copy_data = self.data.copy()
        copy_data["ema_short"] = (
            copy_data["close"].ewm(span=self.short_ema_period, adjust=False).mean()
        )
        copy_data["ema_long"] = (
            copy_data["close"].ewm(span=self.long_ema_period, adjust=False).mean()
        )
        copy_data["macd"] = copy_data["ema_short"] - copy_data["ema_long"]
        copy_data["signal_line"] = (
            copy_data["macd"].ewm(span=self.signal_line_period, adjust=False).mean()
        )

        self.signals["signal"][
            copy_data["macd"] > copy_data["signal_line"]
        ] = 1.0  # Buy signal
        self.signals["signal"][
            copy_data["macd"] < copy_data["signal_line"]
        ] = -1.0  # Sell/short signal

        self.validate_signals()

        del copy_data
