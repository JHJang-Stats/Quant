import pytest
import pandas as pd
from dataset_constructor import MarketData
from strategy.indicator_based import IndicatorBasedStrategy
from strategy.indicator_based import (
    ADLStrategy,
    ADXStrategy,
    BollingerBandsStrategy,
    FibonacciRetracementStrategy,
    IchimokuCloudStrategy,
    MACDStrategy,
    MovingAverageCrossoverStrategy,
    OBVStrategy,
    ParabolicSARStrategy,
    RSIStrategy1,
    RSIStrategy2,
    StochasticOscillatorStrategy,
)
from backtest import Backtest
from forecasting.rolling_forecast import RollingForecast


file_path = "data/crypto/csv/BTC_USDT_4h.csv"
start_date = pd.to_datetime("2019-01-01")
end_date = pd.to_datetime("2019-12-31")


@pytest.fixture
def market_data():
    return MarketData(file_path)


@pytest.fixture
def strategy_classes():
    return [
        ADLStrategy,
        ADXStrategy,
        BollingerBandsStrategy,
        FibonacciRetracementStrategy,
        IchimokuCloudStrategy,
        MACDStrategy,
        MovingAverageCrossoverStrategy,
        OBVStrategy,
        ParabolicSARStrategy,
        RSIStrategy1,
        RSIStrategy2,
        StochasticOscillatorStrategy,
    ]


@pytest.fixture
def strategies(market_data):
    return [strategy_class(market_data.data) for strategy_class in strategy_classes()]


def test_strategies_inheritance(strategy_classes):
    for strategy_class in strategy_classes:
        assert issubclass(
            strategy_class, IndicatorBasedStrategy
        ), f"{strategy_class.__name__} does not inherit from IndicatorBasedStrategy"


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
        ADLStrategy,
        ADXStrategy,
        BollingerBandsStrategy,
        FibonacciRetracementStrategy,
        IchimokuCloudStrategy,
        MACDStrategy,
        MovingAverageCrossoverStrategy,
        OBVStrategy,
        ParabolicSARStrategy,
        RSIStrategy1,
        RSIStrategy2,
        StochasticOscillatorStrategy,
    ]
    return [
        RollingForecast(
            data=market_data.data,
            strategy_class=strategy_class,
            start_date=start_date,
            end_date=end_date,
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
