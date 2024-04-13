import pytest
import pandas as pd
from dataset_constructor import MarketData
from strategy.statistical_model_startegy import StatisticalModelStrategy
from strategy.statistical_model_startegy.autoregressive_model_strategy import (
    ARModelStrategy,
)
from strategy.statistical_model_startegy.arima_model_strategy import ARIMAModelStrategy
from strategy.statistical_model_startegy.sarima_model_strategy import SARIMAModelStrategy
from backtest import Backtest
from forecasting.rolling_forecast import RollingForecast


file_path = "data/crypto/csv/BTC_USDT_4h.csv"
fit_start_date = pd.to_datetime("2018-01-01")
fit_end_date = pd.to_datetime("2019-01-01")
fit_duration = fit_end_date - fit_start_date
start_date = pd.to_datetime("2019-01-01")
end_date = pd.to_datetime("2019-12-31")
predict_period = (start_date, end_date)


@pytest.fixture
def market_data():
    return MarketData(file_path)


@pytest.fixture
def strategy_classes():
    return [
        ARModelStrategy,
        ARIMAModelStrategy,
        SARIMAModelStrategy,
    ]


@pytest.fixture
def strategies(market_data):
    return [
        strategy_class(
            market_data.data, fit_duration=fit_duration, predict_period=predict_period
        )
        for strategy_class in strategy_classes
    ]


def test_strategies_inheritance(strategy_classes):
    for strategy_class in strategy_classes:
        assert issubclass(
            strategy_class, StatisticalModelStrategy
        ), f"{strategy_class.__name__} does not inherit from StatisticalModelStrategy"


def test_strategies_run_without_errors(strategies, market_data):
    for strategy in strategies:
        backtest = Backtest(
            market_data.data,
            strategy,
            start_date=start_date,
            end_date=end_date,
        )

        backtest.run()
        portfolio = backtest.simulate_trades(initial_capital=10000)
        metrics = backtest.calculate_metrics()

        assert portfolio is not None
        assert metrics is not None
        assert (
            portfolio.index == market_data.data[start_date:end_date].index
        ).all(), "Index intervals are not same."


@pytest.fixture
def strategies(market_data):
    strategies_class = [
        ARModelStrategy,
        ARIMAModelStrategy,
        SARIMAModelStrategy,
    ]
    return [
        RollingForecast(
            data=market_data.data,
            strategy_class=strategy_class,
            start_date=start_date,
            end_date=end_date,
            val_ratio=0.2,
            fit_duration=fit_duration,
        )
        for strategy_class in strategies_class
    ]


def test_rolling_forecast_strategies_run_without_errors(strategies, market_data):
    for strategy in strategies:
        backtest = Backtest(market_data.data, strategy)
        backtest.run()
        portfolio = backtest.simulate_trades(initial_capital=10000)
        metrics = backtest.calculate_metrics()

        assert portfolio is not None
        assert metrics is not None
        assert (
            portfolio.index == market_data.data[start_date:end_date].index
        ).all(), "Index intervals are not same."
