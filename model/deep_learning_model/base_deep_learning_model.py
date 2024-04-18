from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DeepLearningModel(ABC):
    """
    Abstract base class for deep learning time series forecasting models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fit_period=(None, None),
        predict_period=(None, None),
        epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        model: Sequential = None,
    ):
        if not (isinstance(fit_period, tuple) and len(fit_period) == 2):
            raise ValueError("fit_period must be a tuple with two elements")
        if not (isinstance(predict_period, tuple) and len(predict_period) == 2):
            raise ValueError("predict_period must be a tuple with two elements")

        self.data = data
        self.fit_period = fit_period
        self.predict_period = predict_period
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
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

        assert self.fit_period[0] is not None
        assert self.fit_period[1] is not None
        assert self.predict_period[0] is not None
        assert self.predict_period[1] is not None

    @abstractmethod
    def build_model(self):
        """
        Build the deep learning model architecture.
        """
        raise NotImplementedError(
            "Subclasses must implement this method to build the model."
        )

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self) -> pd.Series:
        raise NotImplementedError("Subclasses must implement this method")

    def fit_predict(self) -> pd.Series:
        self.fit()
        return self.predict()

    def compile_model(self, optimizer="adam", loss="mean_squared_error"):
        """
        Compile the model with the given optimizer, loss function, and learning rate.
        """
        if self.model is not None:
            # Check if the optimizer is a string and needs to be instantiated with a learning rate
            if isinstance(optimizer, str) and optimizer.lower() == "adam":
                optimizer = Adam(learning_rate=self.learning_rate)
            # Optionally, handle other optimizers similarly if needed

            self.model.compile(optimizer=optimizer, loss=loss)
        else:
            raise ValueError(
                "Model is not built. Ensure `build_model()` is called first."
            )
