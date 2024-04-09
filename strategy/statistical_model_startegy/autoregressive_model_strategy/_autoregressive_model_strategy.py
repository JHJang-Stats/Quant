from ...statistical_model_startegy import StatisticalModelStrategy
from model.statistical_model.autoregressive_model import ARModel


class ARModelStrategy(StatisticalModelStrategy):
    def __init__(
        self,
        data,
        lags=1,
        fit_start_date=None,
        fit_end_date=None,
        predict_steps=1,
        thresholds=0.004,
    ):
        super().__init__(data)
        self.model = ARModel(data, lags, fit_start_date, fit_end_date, predict_steps)
        self.predict_steps = predict_steps
        self.thresholds = thresholds

    def fit_model(self):
        """
        Fits the AR model to the data.
        """
        self.model.fit()
        self.coefficients = self.model.coefficients

    def generate_signals(self):
        if self.coefficients is None:
            self.fit_model()

        super().generate_signals()
        self.signals["signal"][
            self.predictions["Y_hat"]
            > self.predictions["close"] * (1 + self.thresholds)
        ] = 1
        self.signals["signal"][
            self.predictions["Y_hat"]
            < self.predictions["close"] * (1 - self.thresholds)
        ] = -1

        self.validate_signals()

        del self.predictions
