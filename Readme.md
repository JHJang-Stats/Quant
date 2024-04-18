# Project Title: Quant

## Summary
This project implements a scalable backtesting system based on simple trading strategies. It handles cryptocurrency data and includes features that integrate and test various trading strategies and models. Note that this is not an alpha strategy!

## Module Description
### `data_management`
- `fetch_and_save_crypto_data.py`: Manages and collects cryptocurrency data. Please note that downloading data can be time-consuming and should be used with caution.

### `Model`
- Defines models used for executing strategies. Models include `fit` and `predict` methods.

### `Strategy`
- Includes various strategy classes that generate trading signals. This is not an alpha strategy, so please refer to the script and modify it. Currently, most implementations of the strategy are quite simple, comparing the next closing price estimates with the current closing price to generate a signal.
  - `IndicatorBasedStrategy`: Can be implemented with just hyperparameter settings.
  - `StatisticalModelStrategy`: Uses statistical models and requires a training process (`fit`).
  - `DeepLearningModelStrategy`: Utilizes deep learning models.
  - `Ensemble`: Combines multiple models in an ensemble strategy.

### `Backtest`
- A backtesting module to test implemented strategies.

### `RollingForecast`
- Derives optimal hyperparameters for each segment using a rolling window approach.

## Usage Examples
- Refer to the Jupyter notebook files in the `notebooks/` directory for examples of how to use the implemented system.

## Planned Future Implementations
- Portfolio Rebalancing Strategy: Implement a strategy for portfolio rebalancing.
- Get Alternative Data: Utilize alternative data sources.
