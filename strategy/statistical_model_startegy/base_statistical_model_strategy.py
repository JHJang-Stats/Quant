from abc import ABC, abstractmethod
from ..base_strategy import Strategy
import pandas as pd
from datetime import datetime


class StatisticalModelStrategy(Strategy, ABC):
    def __init__(
        self,
        data,
        fit_duration=None,
        predict_period=(None, None),
    ):
        super().__init__(data)
        self.fit_duration = fit_duration
        self.predict_period = predict_period
        self._initialize_signals()

    def _initialize_signals(self):
        self.signals = pd.DataFrame(
            index=self.data[self.predict_period[0] : self.predict_period[1]].index
        )
        self.signals["signal"] = 0

    def generate_signals(self):
        predictions_list = []
        for predict_date in self.data[
            self.predict_period[0] : self.predict_period[1]
        ].index:
            fit_period = (predict_date - self.fit_duration, predict_date)
            self.model.fit_period = fit_period
            self.model.predict_period = (predict_date, predict_date)

            self.model.fit()
            prediction = self.model.predict()
            predictions_list.append(prediction)

        self.predictions = pd.concat(predictions_list)
