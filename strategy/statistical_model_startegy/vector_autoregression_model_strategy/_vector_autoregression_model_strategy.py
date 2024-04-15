from .. import StatisticalModelStrategy
from model.statistical_model.vector_autoregression_model import VARModel


class VARModelStrategy(StatisticalModelStrategy):
    def __init__(
        self,
        data,
        lags=1,
        fit_duration=None,
        predict_period=(None, None),
        thresholds=0.0025,
    ):
        super().__init__(data, fit_duration=fit_duration, predict_period=predict_period)
        self.thresholds = thresholds
        self.model = VARModel(self.data, lags=lags)

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
