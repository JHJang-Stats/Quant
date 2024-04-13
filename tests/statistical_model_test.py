import pytest
import pandas as pd
from model.statistical_model.arima_model import ARIMAModel
from model.statistical_model.autoregressive_model import ARModel
from dataset_constructor import MarketData


@pytest.fixture(scope="module")
def market_data():
    file_path = "data/crypto/csv/BTC_USDT_4h.csv"
    return MarketData(file_path)


@pytest.fixture(params=[ARModel, ARIMAModel])
def model_class(request):
    return request.param


# Test 1: Predictions made for a specific period after fitting the model
def test_predictions_for_specific_period(model_class, market_data):
    fit_start_date = pd.to_datetime("2019-09-01")
    fit_end_date = pd.to_datetime("2019-12-31")
    predict_end_date = pd.to_datetime("2020-01-05")

    fit_period = (fit_start_date, fit_end_date)
    predict_period = (fit_end_date, predict_end_date)

    hyperparms = {
        "fit_period": fit_period,
        "predict_period": predict_period,
    }

    model = model_class(market_data.data, **hyperparms)
    model.fit()
    predictions = model.predict()

    assert (
        predictions.index
        == market_data.data[predict_period[0] : predict_period[1]].index
    ).all()
    assert (
        not predictions.isnull().any().any()
    ), "There are NaN values in the dataframe"


# Test 2: Predictions made up to a specific end date without a defined start date
def test_predictions_up_to_end_date(model_class, market_data):
    fit_start_date = pd.to_datetime("2019-09-01")
    fit_end_date = pd.to_datetime("2019-12-31")
    predict_end_date = pd.to_datetime("2020-01-05")

    fit_period = (fit_start_date, fit_end_date)
    predict_period = (None, predict_end_date)

    hyperparms = {
        "fit_period": fit_period,
        "predict_period": predict_period,
    }

    model = model_class(market_data.data, **hyperparms)
    model.fit()
    predictions = model.predict()

    assert (
        predictions.index == market_data.data[fit_period[1] : predict_period[1]].index
    ).all()


# Test 3: Predictions made without a defined period
def test_predictions_without_defined_period(model_class, market_data):
    fit_start_date = pd.to_datetime("2019-09-01")
    fit_end_date = pd.to_datetime("2019-12-31")

    fit_period = (fit_start_date, fit_end_date)
    predict_period = (None, None)

    hyperparms = {
        "fit_period": fit_period,
        "predict_period": predict_period,
    }

    model = model_class(market_data.data, **hyperparms)
    model.fit()
    predictions = model.predict()

    # This assertion is marked to fail as it compares incompatible types. Needs clarification.
    assert predictions.index == fit_end_date
    assert (
        not predictions.isnull().any().any()
    ), "There are NaN values in the dataframe"
