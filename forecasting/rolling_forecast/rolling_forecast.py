import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from itertools import product
from backtest import Backtest
from datetime import datetime
from strategy import Strategy

from strategy.indicator_based import IndicatorBasedStrategy
from strategy.statistical_model_startegy import StatisticalModelStrategy


class RollingForecast(Strategy):
    def __init__(
        self,
        data,
        strategy_class,
        start_date=None,
        end_date=None,
        rolling_window=365,
        evaluation_window=7,
        hyperparameter_grid=None,
        val_ratio=0,
    ):
        super().__init__(data)
        self.strategy_class = strategy_class
        self.start_date = start_date
        self.end_date = end_date
        self.rolling_window = rolling_window
        self.evaluation_window = evaluation_window
        self.hyperparameter_grid = hyperparameter_grid or {}
        self.best_hyperparams_history = []
        self.portfolio = None
        self.val_ratio = val_ratio

        if issubclass(self.strategy_class, IndicatorBasedStrategy):
            assert self.val_ratio == 0

    def _filter_data(self):
        def process_date(date):
            if isinstance(date, str):
                return pd.to_datetime(date)
            elif isinstance(date, (int, float, np.integer)):
                return pd.to_datetime(date, unit="ms")
            elif isinstance(date, datetime):
                return date
            else:
                raise ValueError(
                    "Date must be a string or a numeric timestamp (int, float)."
                )

        if self.start_date is not None:
            self.start_date = process_date(self.start_date)
        if self.end_date is not None:
            self.end_date = process_date(self.end_date)

        if self.start_date is not None:
            start_idx = max(
                np.where(self.data.index == self.start_date)[0][0]
                - self.rolling_window,
                0,
            )
            self.data = self.data.iloc[start_idx:,]

        if self.end_date is not None:
            self.data = self.data[: self.end_date]

    def _optimize_indicator_based_strategy_hyperparameters(
        self,
        train_start_date,
        train_end_date,
    ):
        best_hyperparams = None
        best_performance = float("-inf")

        for params in product(*self.hyperparameter_grid.values()):
            hyperparams = dict(zip(self.hyperparameter_grid.keys(), params))
            filtered_data = self.data[: train_end_date].copy()
            strategy_instance = self.strategy_class(filtered_data, **hyperparams)
            backtest_instance = Backtest(
                data=self.data,
                strategy=strategy_instance,
                start_date=train_start_date,
                end_date=train_end_date,
            )
            backtest_instance.run()
            backtest_instance.simulate_trades(initial_capital=10000)
            metrics = backtest_instance.calculate_metrics()
            performance_metric = metrics["sharpe_ratio"]

            if performance_metric > best_performance:
                best_performance = performance_metric
                best_hyperparams = hyperparams

            assert len(backtest_instance.portfolio) == self.rolling_window

        return best_hyperparams

    def _optimize_statistical_model_strategy_hyperparameters(
        self, train_start_date, train_end_date, val_start_date, val_end_date
    ):
        best_hyperparams = None
        best_performance = float("-inf")

        for params in product(*self.hyperparameter_grid.values()):
            hyperparams = dict(zip(self.hyperparameter_grid.keys(), params))
            filtered_data = self.data[: val_end_date].copy()
            strategy_instance = self.strategy_class(
                filtered_data,
                fit_start_date=train_start_date,
                fit_end_date=train_end_date,
                predict_start_date=val_start_date,
                predict_end_date=val_end_date,
                **hyperparams
            )
            backtest_instance = Backtest(
                data=self.data,
                strategy=strategy_instance,
                start_date=val_start_date,
                end_date=val_end_date,
            )
            backtest_instance.run()
            backtest_instance.simulate_trades(initial_capital=10000)
            metrics = backtest_instance.calculate_metrics()
            performance_metric = metrics["sharpe_ratio"]

            if performance_metric > best_performance:
                best_performance = performance_metric
                best_hyperparams = hyperparams

            assert len(backtest_instance.portfolio) == len(
                self.data[val_start_date:val_end_date]
            )

        return best_hyperparams

    def _backtest_single_window(self, test_start_index):
        """Function to backtest a single window and return signals."""
        train_start_index = max(test_start_index - self.rolling_window, 0)
        val_end_index = test_start_index - 1
        train_end_index = train_start_index + int(
            (val_end_index - train_start_index) * (1 - self.val_ratio)
        )
        val_start_index = train_end_index
        test_end_index = (
            min(test_start_index + self.evaluation_window, len(self.data)) - 1
        )

        train_start_date = self.data.index[train_start_index]
        train_end_date = self.data.index[train_end_index]
        val_start_date = self.data.index[val_start_index]
        val_end_date = self.data.index[val_end_index]
        test_start_date = self.data.index[test_start_index]
        test_end_date = self.data.index[test_end_index]

        if issubclass(self.strategy_class, IndicatorBasedStrategy):
            best_hyperparams = self._optimize_indicator_based_strategy_hyperparameters(
                train_start_date,
                train_end_date,
            )
            filtered_data = self.data[: test_end_date].copy()
            strategy_instance = self.strategy_class(filtered_data, **best_hyperparams)

        elif issubclass(self.strategy_class, StatisticalModelStrategy):
            best_hyperparams = (
                self._optimize_statistical_model_strategy_hyperparameters(
                    train_start_date, train_end_date, val_start_date, val_end_date
                )
            )
            filtered_data = self.data[: test_end_date].copy()
            strategy_instance = self.strategy_class(
                filtered_data,
                fit_start_date=train_start_date,
                fit_end_date=val_end_date,
                predict_start_date=test_start_date,
                predict_end_date=test_end_date,
                **best_hyperparams
            )

        else:
            raise ValueError

        backtest_instance = Backtest(
            data=self.data,
            strategy=strategy_instance,
            start_date=test_start_date,
            end_date=test_end_date,
        )
        backtest_instance.run()
        signal_df = backtest_instance.signals

        assert len(signal_df) <= self.evaluation_window
        return signal_df

    def generate_signals(self):
        self._filter_data()
        self.initialize_signals()
        self.signals["signal"] = np.nan
        # Use Joblib to parallelize the backtest across rolling windows
        signal_dfs = Parallel(n_jobs=-1)(
            delayed(self._backtest_single_window)(test_start_index)
            for test_start_index in range(
                self.rolling_window, len(self.data), self.evaluation_window
            )
        )

        # Concatenate signals while preserving the order
        concatenated_signals = pd.concat(signal_dfs)
        self.signals.update(concatenated_signals)
        self.signals.dropna(inplace=True)

        self.validate_signals()

    # def generate_signals(self):
    #     self._filter_data()
    #     self.initialize_signals()
    #     self.signals["signal"] = np.nan
    #     signal_dfs = []
    #     for test_start_index in range(
    #         self.rolling_window, len(self.data), self.evaluation_window
    #     ):
    #         signal_df = self._backtest_single_window(test_start_index)
    #         signal_dfs.append(signal_df)

    #     # Concatenate signals while preserving the order
    #     concatenated_signals = pd.concat(signal_dfs)
    #     self.signals.update(concatenated_signals)
    #     self.signals.dropna(inplace=True)
    #     self.validate_signals()
