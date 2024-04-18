import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ...deep_learning_model import DeepLearningModel


class LSTMModel(DeepLearningModel):
    def __init__(
        self,
        data,
        fit_period=(None, None),
        predict_period=(None, None),
        epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        time_step=20,
        lstm_units=50,
        dropout_rate=0.2,
        optimizer="adam",
        loss="mean_squared_error",
    ):
        super().__init__(
            data, fit_period, predict_period, epochs, batch_size, learning_rate
        )
        self.time_step = time_step
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.build_model()  # Build and compile the LSTM model1

    def build_model(self):
        """
        Build the LSTM network architecture.
        """
        self.model = Sequential(
            [
                LSTM(
                    self.lstm_units,
                    return_sequences=True,
                    input_shape=(self.time_step, self.data.shape[1]),
                ),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units),
                Dropout(self.dropout_rate),
                Dense(1),
            ]
        )
        self.compile_model(self.optimizer, self.loss)

    def create_datsets(self, data, period):
        period_data = data[period[0] : period[1]]
        start = data.index.get_loc(period_data.index[0])
        end = data.index.get_loc(period_data.index[-1])
        X = [data.iloc[i - self.time_step : i].values for i in range(start, end + 1)]
        y = data[period[0] : period[1]]["close"].values

        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.time_step
        assert X.shape[2] == self.data.shape[1]

        return X, y

    def fit(self):
        X, y = self.create_datsets(self.data, self.fit_period)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        self.is_fitted = True

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

        # (t+1) time
        start = self.data.index.get_loc(
            self.data[self.predict_period[0] : self.predict_period[1]].index[0]
        )
        end = self.data.index.get_loc(
            self.data[self.predict_period[0] : self.predict_period[1]].index[-1]
        )
        predict_period = (self.data.index[start + 1], self.data.index[end + 1])

        X, y = self.create_datsets(self.data, predict_period)
        predictions = self.model.predict(X).flatten()
        predictions = pd.DataFrame(predictions, columns=["Y_hat^(t+1)"])
        predictions.index = self.data[
            self.predict_period[0] : self.predict_period[1]
        ].index
        merged_df = self.data.join(predictions)[["Y_hat^(t+1)", "close"]]
        merged_df = merged_df[self.predict_period[0] : self.predict_period[1]]
        assert (
            not merged_df.isnull().any().any()
        ), "There are NaN values in the dataframe"

        return merged_df
