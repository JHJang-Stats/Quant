import numpy as np
import pandas as pd
from numpy.linalg import inv
from ...statistical_model import StatisticalModel
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore", category=Warning)


class ARIMAModel(StatisticalModel):
    def __init__(
        self,
        data,
        order=(1, 0, 0),  # order (p,d,q) for ARIMA modoel
        fit_period=(None, None),
        predict_period=(None, None),
    ):
        super().__init__(data, fit_period, predict_period)
        self.order = order

    def fit(self):
        filtered_data = self._filter_data_for_fit_period()
        # freq = filtered_data.index.to_series().diff().mode().iloc[0]
        self.model = ARIMA(filtered_data["close"], order=self.order)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        del filtered_data

    def predict(self) -> pd.DataFrame:
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
            name="predicted_mean"
        )
        merged_df = self.data.join(predictions)[["predicted_mean", "close"]]
        shifted_predicted_mean = merged_df["predicted_mean"].shift(-1)
        merged_df["predicted_mean"] = shifted_predicted_mean
        merged_df = merged_df[self.predict_period[0] : self.predict_period[1]]
        merged_df.rename(columns={"predicted_mean": "Y_hat^(t+1)"}, inplace=True)

        return merged_df
