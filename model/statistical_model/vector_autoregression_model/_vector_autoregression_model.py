import numpy as np
import pandas as pd
from numpy.linalg import inv
import logging
from ...statistical_model import StatisticalModel
from statsmodels.tsa.vector_ar.var_model import VAR

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VARModel(StatisticalModel):
    def __init__(
        self,
        data,
        lags=1,
        fit_period=(None, None),
        predict_period=(None, None),
        columns=None,
    ):
        super().__init__(data, fit_period, predict_period)
        self.lags = lags
        self.columns = columns if columns is not None else list(data.columns)
        self.coefficients = None
        self._validate_columns()

    def _validate_columns(self):
        missing_columns = [col for col in self.columns if col not in self.data.columns]
        if missing_columns:
            logger.warning(
                f"Columns {missing_columns} not found in data. They will be ignored."
            )
        self.columns = [col for col in self.columns if col not in missing_columns]

    def fit(self):
        if not self.columns:
            logger.error("No valid columns available for fitting the model.")
            return
        filtered_data = self._filter_data_for_fit_period()[self.columns]
        inferred_freq = pd.infer_freq(filtered_data.index)

        if inferred_freq:
            filtered_data.index = filtered_data.index.to_period(inferred_freq)
        self.model = VAR(
            filtered_data,
        )
        self.fitted_model = self.model.fit(maxlags=self.lags)
        self.is_fitted = True
        self.filtered_data = filtered_data

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
        if not self.columns:
            logger.error("No valid columns available for prediction.")
            return pd.DataFrame()

        steps = len(self.data[self.fit_period[1] : self.predict_period[1]])
        predictions = self.fitted_model.forecast(
            y=self.filtered_data.values[-self.lags :], steps=steps
        )
        predictions = pd.DataFrame(predictions)
        predictions.index = self.data[self.predict_period[0] :].index[:steps]
        predictions.columns = self.filtered_data.columns
        predictions = predictions[["close"]]
        predictions = predictions.rename(columns={"close": "Y_hat^(t+1)"})
        merged_df = self.data.join(predictions)[["Y_hat^(t+1)", "close"]]
        merged_df = merged_df[self.predict_period[0] : self.predict_period[1]]
        assert (
            not merged_df.isnull().any().any()
        ), "There are NaN values in the dataframe"

        return merged_df
