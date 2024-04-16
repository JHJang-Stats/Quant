from .. import StatisticalModelStrategy
from model.statistical_model.ets_model import ETSModel


class ETSModelStrategy(StatisticalModelStrategy):
    def __init__(
        self,
        data,
        error="mul",
        trend="mul",
        damped_trend=True,
        seasonal=None,
        seasonal_periods=None,
        fit_duration=None,
        predict_period=(None, None),
        thresholds=0.0025,
    ):
        super().__init__(data, fit_duration=fit_duration, predict_period=predict_period)
        self.thresholds = thresholds
        self.model = ETSModel(
            self.data,
            error=error,
            trend=trend,
            damped_trend=damped_trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
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
