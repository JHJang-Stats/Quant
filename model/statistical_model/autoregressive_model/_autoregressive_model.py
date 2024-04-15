import numpy as np
import pandas as pd
from numpy.linalg import inv
from ...statistical_model import StatisticalModel


class ARModel(StatisticalModel):
    def __init__(
        self,
        data,
        lags=1,
        fit_period=(None, None),
        predict_period=(None, None),
    ):
        super().__init__(data, fit_period, predict_period)
        self.lags = lags
        self.coefficients = None

    def fit(self):
        filtered_data = self._filter_data_for_fit_period()
        Y = filtered_data["close"][self.lags :].values
        X = [
            filtered_data["close"].shift(i)[self.lags :].values
            for i in range(1, self.lags + 1)
        ]
        X = np.column_stack(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.coefficients = inv(X.T @ X) @ X.T @ Y
        self.is_fitted = True

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

        predict_start_date, predict_end_date = self.predict_period

        X = [
            self.data["close"].shift(i)[self.lags :].values for i in range(0, self.lags)
        ]
        X = np.column_stack(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X = pd.DataFrame(X)
        X.index = self.data.index[self.lags :]
        Y_hat = X @ self.coefficients

        Y_hat = Y_hat[predict_start_date:predict_end_date]

        Y_hat = Y_hat.to_frame(name="Y_hat^(t+1)")
        merged_df = pd.merge(
            Y_hat, self.data["close"], left_index=True, right_index=True, how="left"
        )

        return merged_df
