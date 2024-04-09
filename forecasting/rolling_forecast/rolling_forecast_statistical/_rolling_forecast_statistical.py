import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from itertools import product
from backtest import Backtest
from datetime import datetime
from strategy import Strategy
from .. import RollingForecast


class RollingForecastStatistical(RollingForecast):
    def __init__(
        self,
        data,
        strategy_class,
        start_date=None,
        end_date=None,
        rolling_window=365,
        evaluation_window=7,
        hyperparameter_grid=None,
        val_ratio=0.2,
    ):
        super().__init__(
            data,
            strategy_class,
            start_date=None,
            end_date=None,
            rolling_window=365,
            evaluation_window=7,
            hyperparameter_grid=None,
        )
        self.val_ratio = val_ratio

    def _filter_data(self):
        super()._filter_data()

    def _optimize_hyperparameters(self, train_start_date, train_end_date):
        raise NotImplementedError

    def _backtest_single_window(self, test_start_index):
        raise NotImplementedError

    def generate_signals(self):
        super().generate_signals()
