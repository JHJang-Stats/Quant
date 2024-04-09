import pandas as pd
import numpy as np
from datetime import datetime


class Backtest:
    def __init__(
        self,
        data,
        strategy,
        start_date=None,
        end_date=None,
        fee=2e-4,
        leverage=1,
        enable_logging=False,
    ):
        self.data = data
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.fee = fee
        self.leverage = leverage
        self.signals = None
        self.portfolio = None
        self.metrics = {}
        self.enable_logging = enable_logging  # Add enable_logging parameter
        self._filter_data()

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

        if self.end_date is not None:
            self.data = self.data[self.data.index <= self.end_date]

    def run(self):
        self.strategy.data = self.data
        self.strategy.initialize_signals()
        self.strategy.generate_signals()
        self.signals = self.strategy.signals
        if self.start_date is not None:
            self.signals = self.signals[self.signals.index >= self.start_date]

    def simulate_trades(self, initial_capital=10000):
        self.portfolio = pd.DataFrame(index=self.signals.index)
        self.portfolio["signal"] = self.signals["signal"]
        self.portfolio["change_signal"] = self.portfolio["signal"].diff()
        self.portfolio.at[self.portfolio.index[0], "change_signal"] = self.portfolio[
            "signal"
        ].iloc[0]

        self.portfolio = self.portfolio.join(self.data["close"])
        self.portfolio["entry_price"] = None

        self.portfolio["contract_numbers"] = 0
        self.portfolio["cash"] = initial_capital
        self.portfolio["total"] = initial_capital

        prev_signal = 0
        prev_entry_price = None
        prev_cash = initial_capital
        prev_contract_numbers = 0
        flag = 0

        for idx, row in self.portfolio.iterrows():
            assert flag < 2
            # position 신규 진입
            if prev_signal == 0:
                if row["change_signal"] != 0:
                    if row["signal"] == 0:
                        flag += 1
                    else:
                        # 보유 현금 전부다 self.leverage 로 진입
                        prev_signal = row["signal"]
                        prev_entry_price = row["close"] * (1 + prev_signal * self.fee)
                        prev_contract_numbers = (
                            prev_cash / prev_entry_price * self.leverage * prev_signal
                        )  # contract number 음수일시 sell position
                        prev_cash = (1 - self.leverage * prev_signal) * prev_cash

            # position 존재할 때
            else:
                if row["change_signal"] == 0:
                    # portfolio value 만 바뀜
                    pass
                else:
                    # 청산 진행 (청산시 수수료 고려)
                    prev_entry_price = None
                    prev_cash = prev_cash + self.portfolio.at[
                        idx, "close"
                    ] * prev_contract_numbers * (1 - prev_signal * self.fee)
                    prev_signal = 0
                    prev_contract_numbers = 0

                    if row["signal"] != 0:
                        # 보유 현금 전부 다시 self.leverage 로 진입
                        prev_signal = row["signal"]
                        prev_entry_price = row["close"] * (1 + prev_signal * self.fee)
                        prev_contract_numbers = (
                            prev_cash / prev_entry_price * self.leverage * prev_signal
                        )  # contract number 음수일시 sell position
                        prev_cash = (1 - self.leverage * prev_signal) * prev_cash

            self.portfolio.at[idx, "entry_price"] = prev_entry_price
            self.portfolio.at[idx, "contract_numbers"] = prev_contract_numbers
            self.portfolio.at[idx, "cash"] = prev_cash
            self.portfolio.at[idx, "total"] = (
                prev_cash + self.portfolio.at[idx, "close"] * prev_contract_numbers
            )

        if self.enable_logging:
            self._log_position_changes()

        return self.portfolio

    def _log_position_changes(self):
        """
        Logs every time there is a change in change_signal with additional portfolio information.
        """
        changes = self.portfolio[self.portfolio["change_signal"] != 0]

        for date, change in changes.iterrows():
            # date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
            position_type = (
                "Long"
                if change["signal"] > 0
                else "Short"
                if change["signal"] < 0
                else "Flat"
            )
            entry_price = (
                f"{change['entry_price']:.4f}"
                if change["entry_price"] is not None
                else "N/A"
            )
            log_message = (
                f"Date: {date}, Position: {position_type}, "
                f"Entry Price: {entry_price}, "
                f"Market Price: {change['close']:.4f}, "
                f"Contract Numbers: {change['contract_numbers']:.2f}, "
                f"Cash: {change['cash']:.2f}, "
                f"Portfolio Value: {change['total']:.2f}"
            )
            print(log_message)

    def calculate_metrics(self, risk_free_rate=0.0):
        if self.portfolio is None:
            raise Exception("Portfolio not initialized. Run simulate_trades first.")

        # Ensure the index is a DateTimeIndex to calculate time differences
        if not isinstance(self.portfolio.index, pd.DatetimeIndex):
            self.portfolio.index = pd.to_datetime(self.portfolio.index)

        self.portfolio["portfolio_returns"] = self.portfolio["total"].pct_change()
        self.portfolio[
            "time_diff"
        ] = self.portfolio.index.to_series().diff().dt.total_seconds() / (60 * 60 * 24)
        self.portfolio["excess_returns"] = self.portfolio[
            "portfolio_returns"
        ] - risk_free_rate / (365 / self.portfolio["time_diff"])
        excess_returns = self.portfolio["excess_returns"].dropna()
        avg_intervals_per_year = 365 / self.portfolio["time_diff"].mean()

        # Calculate the Sharpe Ratio with dynamic annualization factor
        if excess_returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (
                excess_returns.mean()
                / excess_returns.std()
                * np.sqrt(avg_intervals_per_year)
            )

        # Maximum Drawdown calculation as before
        cumulative_returns = (1 + excess_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()

        self.metrics = {"sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown}

        # Optionally clean up temporary columns
        self.portfolio.drop(
            columns=["portfolio_returns", "time_diff", "excess_returns"], inplace=True
        )

        return self.metrics
