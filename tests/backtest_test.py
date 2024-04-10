import pytest
from dataset_constructor import MarketData
from strategy.vanilla import VanillaStrategy
from backtest import Backtest

# Define the file paths to test
file_paths = [
    "data/crypto/csv/BTC_USDT_1d.csv",
    "data/crypto/csv/BTC_USDT_4h.csv",
    "data/crypto/csv/BTC_USDT_1h.csv",
    "data/crypto/csv/BTC_USDT_15m.csv",
]

start_date = "2018-01-01"
end_date = None


@pytest.fixture(scope="module")
def baseline_metrics():
    # Load the baseline data, run the strategy and backtest, then calculate metrics
    market_data = MarketData(file_paths[0])
    strategy = VanillaStrategy(market_data.data, start_date=start_date)
    backtest = Backtest(
        market_data.data,
        strategy,
        start_date=start_date,
        end_date=end_date,
        fee=2e-4,
        enable_logging=True,
    )
    backtest.run()
    portfolio = backtest.simulate_trades(initial_capital=10000)
    metrics = backtest.calculate_metrics()
    return metrics, portfolio["total"].iloc[-1]


@pytest.mark.parametrize("file_path", file_paths)
def test_strategy_metrics(file_path, baseline_metrics):
    baseline_metric, baseline_total = baseline_metrics

    # Load the market data, run the strategy and backtest, then calculate metrics for the current file path
    market_data = MarketData(file_path)
    strategy = VanillaStrategy(market_data.data, start_date=start_date)
    backtest = Backtest(
        market_data.data,
        strategy,
        start_date=start_date,
        end_date=end_date,
        fee=2e-4,
        enable_logging=True,
    )
    backtest.run()
    portfolio = backtest.simulate_trades(initial_capital=10000)
    metrics = backtest.calculate_metrics()

    # Verify the metrics and portfolio total are within 2% of the baseline
    for metric in ["sharpe_ratio", "max_drawdown"]:
        assert (
            abs(metrics[metric] - baseline_metric[metric]) / baseline_metric[metric]
            <= 0.02
        ), f"{metric} not within 2% of baseline"

    final_total = portfolio["total"].iloc[-1]
    assert (
        abs(final_total - baseline_total) / baseline_total <= 0.02
    ), "Final total not within 2% of baseline"
