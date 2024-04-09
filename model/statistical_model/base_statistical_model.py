from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime


class StatisticalModel(ABC):
    """
    Abstract base class for traditional statistical time series forecasting models.
    """
    def __init__(self, data: pd.DataFrame, fit_start_date=None, fit_end_date=None, predict_steps=1):
        self.data = data
        self.fit_start_date = fit_start_date
        self.fit_end_date = fit_end_date
        self.predict_steps = predict_steps
        self.is_fitted = False  # Tracks whether the model has been fitted
        self._process_fit_date()

    def _process_fit_date(self):
        def process_date(date):
            if isinstance(date, str):
                return pd.to_datetime(date)
            elif isinstance(date, (int, float, np.integer)):
                return pd.to_datetime(date, unit='ms')
            elif isinstance(date, datetime):
                return date
            else:
                raise ValueError("Date must be a string or a numeric timestamp (int, float).")

        if self.fit_start_date is None:
            self.fit_start_date = self.data.index[0]
        else:
            self.fit_start_date = process_date(self.fit_start_date)
        
        if self.fit_end_date is None:
            self.fit_end_date = self.data.index[-1]
        else:
            self.fit_end_date = process_date(self.fit_end_date)     

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self) -> pd.Series:
        raise NotImplementedError("Subclasses must implement this method")

    def fit_predict(self) -> pd.Series:
        self.fit()
        return self.predict()
