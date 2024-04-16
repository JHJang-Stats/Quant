import pandas as pd
from .. import StatisticalModel
from prophet import Prophet
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True


class ProphetModel(StatisticalModel):
    def __init__(
        self,
        data,
        growth="linear",
        fit_period=(None, None),
        predict_period=(None, None),
    ):
        super().__init__(data, fit_period, predict_period)
        self.growth = growth
        self.validate_inputs()

    def validate_inputs(self):
        if self.growth not in ("linear", "flat"):
            raise ValueError('Parameter "growth" should be "linear" or "flat".')

        self.freq = pd.infer_freq(self.data.index)
        if self.freq is None:
            differences = self.data.index.to_series().diff().dropna()
            most_common_freq = differences.value_counts().idxmax()
            self.freq = most_common_freq
        if self.freq is None:
            raise ValueError("Frequency cannot be determined from the data.")

    def fit(self):
        filtered_data = self._filter_data_for_fit_period()
        df_prophet = pd.DataFrame(
            {
                "ds": filtered_data.index,
                "y": filtered_data["close"],
            }
        )
        # TODO: check `Prophet` hyperparameter
        self.model = Prophet(growth=self.growth)
        self.fitted_model = self.model.fit(df_prophet)
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
        future = self.model.make_future_dataframe(periods=steps, freq=self.freq)
        predictions = self.model.predict(future)[["ds", "yhat"]]
        predictions.rename(
            columns={"ds": "timestamp", "yhat": "Y_hat^(t+1)"}, inplace=True
        )
        predictions.index = predictions["timestamp"]
        del predictions["timestamp"]
        predictions = predictions.shift(-1)

        merged_df = self.data.join(predictions)[["Y_hat^(t+1)", "close"]].dropna()
        merged_df = merged_df[self.predict_period[0] : self.predict_period[1]]

        assert (
            not merged_df.isnull().any().any()
        ), "There are NaN values in the dataframe"

        return merged_df
