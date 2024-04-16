import pandas as pd
from ...statistical_model import StatisticalModel
from statsmodels.tsa.arima.model import ARIMA

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
        inferred_freq = pd.infer_freq(filtered_data.index)
        # TODO: The `inferred_freq` occurs as None when the index interval is not constant.
        # In this case, the SARIMAX model generates the following warning msg.
        # ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
        # FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
        # To get rid of this warning, you need to collect data without missing data.
        # Please add the missing data later.
        if inferred_freq:
            filtered_data.index = filtered_data.index.to_period(inferred_freq)
        self.model = ARIMA(
            filtered_data["close"],
            order=self.order,
            # missing="drop",
        )
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
