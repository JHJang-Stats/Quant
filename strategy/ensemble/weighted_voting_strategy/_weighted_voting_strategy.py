import pandas as pd
import numpy as np
from ...base_strategy import Strategy

class WeightedVotingStrategy(Strategy):
    """
    An ensemble strategy that takes multiple strategies as input along with their respective
    weights and decides the final signal based on weighted voting among the individual
    strategies' signals.

    Attributes:
    - strategies (list): A list of strategy instances.
    - weights (list): A list of weights corresponding to each strategy.
    - data (DataFrame): The market data, assumed to be common across strategies.
    """
    def __init__(self, data, strategies, weights=None):
        super().__init__(data)
        self.strategies = strategies
        self.weights = weights
        if self.weights is None:
            self.weights = [1] * len(strategies)
        assert len(strategies) == len(self.weights), "Each strategy must have a corresponding weight."

    def generate_signals(self):
        # Ensure all strategies have generated their signals and weights are normalized
        for strategy in self.strategies:
            if strategy.signals is None:
                strategy.generate_signals()

        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize weights

        # Initialize an empty DataFrame to store weighted signals
        weighted_signals = pd.DataFrame(index=self.data.index)
        
        # Collect and weight signals from each strategy
        for i, (strategy, weight) in enumerate(zip(self.strategies, self.weights), start=1):
            weighted_signals[f'strategy_{i}'] = strategy.signals['signal'] * weight

        self.signals['signal'] = weighted_signals.sum(axis=1)
        self.signals['signal'] = self.signals['signal'].apply(np.sign)

        self.validate_signals()