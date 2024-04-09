import numpy as np
import pandas as pd
from ...base_strategy import Strategy


class FibonacciRetracementStrategy(Strategy):
    def __init__(self, data, lookback_period=50, ma_period=20):
        super().__init__(data)
        self.lookback_period = lookback_period
        self.ma_period = ma_period

    def generate_signals(self):
        copy_data = self.data.copy()
        copy_data["ma"] = copy_data["close"].rolling(window=self.ma_period).mean()

        # Identify recent high and low for Fibonacci levels
        copy_data["recent_high"] = (
            copy_data["high"].rolling(window=self.lookback_period).max()
        )
        copy_data["recent_low"] = (
            copy_data["low"].rolling(window=self.lookback_period).min()
        )

        # Calculate Fibonacci Retracement Levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            copy_data[f"fib_{int(level*1000)}"] = (
                copy_data["recent_high"]
                - (copy_data["recent_high"] - copy_data["recent_low"]) * level
            )

        # Trend direction
        copy_data["trend"] = np.where(copy_data["close"] > copy_data["ma"], 1, -1)

        # Initialize signal
        self.signals["signal"] = np.nan

        # Define entry conditions
        self.signals.loc[
            (copy_data["close"] > copy_data["fib_618"])
            & (copy_data["close"].shift(1) <= copy_data["fib_618"].shift(1))
            & (copy_data["trend"] == 1),
            "signal",
        ] = 1
        self.signals.loc[
            (copy_data["close"] < copy_data["fib_382"])
            & (copy_data["close"].shift(1) >= copy_data["fib_382"].shift(1))
            & (copy_data["trend"] == -1),
            "signal",
        ] = -1

        # Exit long and short positions
        for level in [level for level in fib_levels if level not in (0.618, 0.382)]:
            self.signals.loc[
                (
                    copy_data[f"fib_{int(level*1000)}"].between(
                        copy_data["low"], copy_data["high"]
                    )
                ),
                "signal",
            ] = 0
            self.signals.loc[
                (
                    copy_data[f"fib_{int(level*1000)}"].between(
                        copy_data["low"], copy_data["high"]
                    )
                ),
                "signal",
            ] = 0

        # TODO: For those with copy_data['signal'] = nan, if 'ma' is inside 'low' and 'high' prices, then copy_data['signal']=0.
        self.signals.loc[
            copy_data["ma"].between(copy_data["low"], copy_data["high"])
            & self.signals["signal"].isna(),
            "signal",
        ] = 0

        # Clean-up copy_data
        self.signals["signal"] = self.signals["signal"].fillna(method="ffill").fillna(0)

        self.validate_signals()

        del copy_data
