import pandas as pd
from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0

    @abstractmethod
    def generate_signals(self):
        raise NotImplementedError("Should implement generate_signals()")

    def validate_signals(self):
        if self.signals is None:
            raise ValueError("self.signals is not set. Please ensure generate_signals generates a signals DataFrame.")
        if not isinstance(self.signals, pd.DataFrame):
            raise TypeError("self.signals must be a pandas DataFrame.")
        if len(self.signals.columns) != 1:
            raise ValueError("self.signals must contain exactly one signal column.")
        if 'signal' not in self.signals.columns:
            raise ValueError("self.signals must contain a 'signal' column.")