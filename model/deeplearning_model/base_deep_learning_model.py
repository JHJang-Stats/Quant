from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime


class DeepLearningModel(ABC):
    """
    Abstract base class for traditional statistical time series forecasting models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fit_period=(None, None),
        predict_period=(None, None),
    ):
        if not (isinstance(fit_period, tuple) and len(fit_period) == 2):
            raise ValueError("fit_period must be a tuple with two elements")

        if not (isinstance(predict_period, tuple) and len(predict_period) == 2):
            raise ValueError("predict_period must be a tuple with two elements")

        self.data = data
        self.fit_period = fit_period
        self.predict_period = predict_period
        self.is_fitted = False  # Tracks whether the model has been fitted
        self._process_fit_and_predict_period()

    def _process_fit_and_predict_period(self):
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

        self.fit_period = (
            process_date(self.fit_period[0])
            if self.fit_period[0] is not None
            else self.data.index[0],
            process_date(self.fit_period[1])
            if self.fit_period[1] is not None
            else self.data.index[-1],
        )

        predict_start = (
            process_date(self.predict_period[0])
            if self.predict_period[0] is not None
            else self.fit_period[1]
        )
        predict_end = (
            process_date(self.predict_period[1])
            if self.predict_period[1] is not None
            else predict_start
        )
        self.predict_period = (predict_start, predict_end)

    def _filter_data_for_fit_period(self):
        fit_start_date, fit_end_date = self.fit_period
        filtered_data = self.data.copy()
        if fit_start_date:
            filtered_data = filtered_data[fit_start_date:]
        if fit_end_date:
            filtered_data = filtered_data[:fit_end_date]
        return filtered_data

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self) -> pd.Series:
        raise NotImplementedError("Subclasses must implement this method")

    def fit_predict(self) -> pd.Series:
        self.fit()
        return self.predict()
