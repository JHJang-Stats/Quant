import pandas as pd
from .. import StatisticalModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as SM_ETSModel
import logging

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ETSModel(StatisticalModel):
    def __init__(
        self,
        data,
        error="mul",
        trend="mul",
        damped_trend=True,
        seasonal=None,
        seasonal_periods=None,
        fit_period=(None, None),
        predict_period=(None, None),
    ):
        super().__init__(data, fit_period, predict_period)
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.validate_inputs()

    def validate_inputs(self):
        if self.error not in ["add", "mul"]:
            raise ValueError(
                f"Invalid error type: {self.error}. Choose 'add' or 'mul'."
            )

        if self.trend not in ["add", "mul", None]:
            raise ValueError(
                f"Invalid trend type: {self.trend}. Choose 'add', 'mul', or None."
            )

        if self.trend is None:
            if self.damped_trend is not None:
                logging.info("Damped trend reset to None due to no trend component.")
                self.damped_trend = False

        if not isinstance(self.damped_trend, bool):
            raise ValueError("damped_trend must be a boolean.")

        if self.seasonal not in ["add", "mul", None]:
            raise ValueError(
                f"Invalid seasonal type: {self.seasonal}. Choose 'add', 'mul', or None."
            )

        if self.seasonal is not None:
            if not isinstance(self.seasonal_periods, int) or self.seasonal_periods <= 0:
                raise ValueError(
                    "seasonal_periods must be a positive integer when seasonal is specified."
                )

    def fit(self):
        filtered_data = self._filter_data_for_fit_period()
        inferred_freq = pd.infer_freq(filtered_data.index)
        if inferred_freq:
            filtered_data.index = filtered_data.index.to_period(inferred_freq)
        self.model = SM_ETSModel(
            filtered_data["close"],
            error=self.error,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self.fitted_model = self.model.fit(disp=0)
        self.is_fitted = True
        del filtered_data

    def predict(self):
        """
        DataFrame
        index: timestamp
        column1: Y_hat^(t+1)
        column2: close
        """
        if not self.is_fitted:
            raise Exception(
                "Model has not been fitted. Please call 'fit' before 'predict'."
            )

        steps = len(self.data[self.fit_period[1] : self.predict_period[1]])
        predictions = self.fitted_model.forecast(steps=steps).to_frame(
            name="Y_hat^(t+1)"
        )
        predictions.index = self.data[
            self.predict_period[0] : self.predict_period[1]
        ].index
        merged_df = self.data.join(predictions)[["Y_hat^(t+1)", "close"]]
        merged_df = merged_df[self.predict_period[0] : self.predict_period[1]]
        assert (
            not merged_df.isnull().any().any()
        ), "There are NaN values in the dataframe"

        return merged_df
