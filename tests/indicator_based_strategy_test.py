import pytest
import pandas as pd
from dataset_constructor import MarketData
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
start_date = "2019-01-01"
end_date = "2020-12-31"


@pytest.fixture
def market_data():
    return MarketData(file_path)


@pytest.fixture
def strategies(market_data):
    return [
        ADLStrategy(market_data.data),
        ADXStrategy(market_data.data),
        BollingerBandsStrategy(market_data.data),
        FibonacciRetracementStrategy(market_data.data),
        IchimokuCloudStrategy(market_data.data),
        MACDStrategy(market_data.data),
        MovingAverageCrossoverStrategy(market_data.data),
        OBVStrategy(market_data.data),
        ParabolicSARStrategy(market_data.data),
        RSIStrategy1(market_data.data),
        RSIStrategy2(market_data.data),
        StochasticOscillatorStrategy(market_data.data),
    ]


def test_strategies_run_without_errors(strategies, market_data):
    for strategy in strategies:
        backtest = Backtest(
            market_data.data, strategy, start_date=start_date, end_date=end_date
        )

        backtest.run()
        portfolio = backtest.simulate_trades(initial_capital=10000)
        metrics = backtest.calculate_metrics()

        assert portfolio is not None
        assert metrics is not None

        assert pd.Timestamp(start_date) <= portfolio.index.min()
        assert pd.Timestamp(end_date) >= portfolio.index.max()


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

        assert pd.Timestamp(start_date) <= portfolio.index.min()
        assert pd.Timestamp(end_date) >= portfolio.index.max()
