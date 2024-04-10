import numpy as np
import pandas as pd
from ...indicator_based import IndicatorBasedStrategy


class ParabolicSARStrategy(IndicatorBasedStrategy):
    def __init__(self, data, initial_af=0.02, step_af=0.02, max_af=0.2):
        super().__init__(data)
        self.initial_af = initial_af  # Initial acceleration factor
        self.step_af = step_af  # Step
        self.max_af = max_af  # Maximum acceleration factor

    def generate_signals(self):
        copy_data = self.data.copy()
        # Initialize the columns with default values
        sar = np.zeros(len(copy_data))
        ep = np.zeros(len(copy_data))
        af = np.zeros(len(copy_data))
        trend = np.zeros(len(copy_data))

        # Set initial values
        sar[0] = (
            copy_data["low"][0]
            if copy_data["close"][0] > copy_data["open"][0]
            else copy_data["high"][0]
        )
        ep[0] = (
            copy_data["high"][0]
            if copy_data["close"][0] > copy_data["open"][0]
            else copy_data["low"][0]
        )
        af[0] = self.initial_af
        trend[0] = 1 if copy_data["close"][0] > copy_data["open"][0] else -1

        for i in range(1, len(copy_data)):
            if trend[i - 1] == 1:  # Upward trend
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                if copy_data["low"][i] < sar[i]:
                    trend[i] = -1  # Change to downward trend
                    sar[i] = ep[i - 1]  # Reset SAR to EP
                    af[i] = self.initial_af  # Reset acceleration factor
                else:
                    if copy_data["high"][i] > ep[i - 1]:
                        ep[i] = copy_data["high"][i]
                        af[i] = min(af[i - 1] + self.step_af, self.max_af)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:  # Downward trend
                sar[i] = sar[i - 1] - af[i - 1] * (sar[i - 1] - ep[i - 1])
                if copy_data["high"][i] > sar[i]:
                    trend[i] = 1  # Change to upward trend
                    sar[i] = ep[i - 1]  # Reset SAR to EP
                    af[i] = self.initial_af  # Reset acceleration factor
                else:
                    if copy_data["low"][i] < ep[i - 1]:
                        ep[i] = copy_data["low"][i]
                        af[i] = min(af[i - 1] + self.step_af, self.max_af)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]

        # Assign calculated arrays back to the DataFrame
        copy_data["sar"] = sar
        copy_data["ep"] = ep
        copy_data["af"] = af
        copy_data["trend"] = trend

        # Generate signals based on the calculated SAR
        self.signals["signal"] = 0  # Initialize signal column
        self.signals.loc[copy_data["close"] > copy_data["sar"], "signal"] = 1
        self.signals.loc[copy_data["close"] < copy_data["sar"], "signal"] = -1

        self.validate_signals()

        del copy_data
