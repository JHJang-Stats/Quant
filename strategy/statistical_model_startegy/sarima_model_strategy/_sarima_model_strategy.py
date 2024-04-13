from .. import StatisticalModelStrategy
from model.statistical_model.sarima_model import SARIMAModel


class SARIMAModelStrategy(StatisticalModelStrategy):
    def __init__(
        self,
        data,
        order=(1, 0, 0),  # order (p,d,q) for ARIMA modoel
        seasonal_order=(0, 0, 0, 0),  # seasonal order (P, D, Q, s) for SARIMA model
        fit_duration=None,
        predict_period=(None, None),
        thresholds=0.0025,
    ):
        super().__init__(data, fit_duration=fit_duration, predict_period=predict_period)
        self.thresholds = thresholds
        self.model = SARIMAModel(
            self.data,
            order=order,
            seasonal_order=seasonal_order,
        )

    def generate_signals(self):
        super().generate_signals()
        self.signals["signal"][
            self.predictions["Y_hat^(t+1)"]
            > self.predictions["close"] * (1 + self.thresholds)
        ] = 1
        self.signals["signal"][
            self.predictions["Y_hat^(t+1)"]
            < self.predictions["close"] * (1 - self.thresholds)
        ] = -1

        self.validate_signals()
        del self.predictions
