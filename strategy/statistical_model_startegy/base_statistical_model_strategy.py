from abc import ABC, abstractmethod
from ..base_strategy import Strategy
import pandas as pd


class StatisticalModelStrategy(Strategy, ABC):
    def __init__(self, data):
        super().__init__(data)
        self.model = None
        self.coefficients = None

    @abstractmethod
    def fit_model(self):
        raise NotImplementedError("Should implement fit_model()")

    def generate_signals(self):
        self.predictions = self.model.predict()
        self.signals = pd.DataFrame(index=self.predictions.index)
        self.signals["signal"] = 0
