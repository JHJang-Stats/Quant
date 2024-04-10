from ..base_strategy import Strategy
import pandas as pd


class IndicatorBasedStrategy(Strategy):
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        super().generate_signals()
