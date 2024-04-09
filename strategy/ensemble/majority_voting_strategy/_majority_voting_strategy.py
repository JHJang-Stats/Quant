import pandas as pd
import numpy as np
from ...base_strategy import Strategy

class MajorityVotingStrategy(Strategy):
    """
    An ensemble strategy that takes multiple strategies as input and decides the final
    signal based on majority voting among the individual strategies' signals.

    Attributes:
    - strategies (list): A list of strategy instances.
    - data (DataFrame): The market data, assumed to be common across strategies.
    """
    def __init__(self, data, strategies):
        super().__init__(data)
        self.strategies = strategies

    def generate_signals(self):
        # Ensure all strategies have generated their signals
        for strategy in self.strategies:
            if strategy.signals is None:
                strategy.generate_signals()

        # Initialize an empty DataFrame to store all signals
        all_signals = pd.DataFrame(index=self.data.index)
        
        # Collect signals from each strategy
        for i, strategy in enumerate(self.strategies, start=1):
            all_signals[f'strategy_{i}'] = strategy.signals['signal']

        self.signals['signal'] = all_signals.mode(axis=1)[0]  # mode() finds the most frequent value
        self.signals['signal'].fillna(0, inplace=True)  # Default to neutral if no majority

        self.validate_signals()

