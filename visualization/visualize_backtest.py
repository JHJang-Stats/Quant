import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def visualize_backtest(portfolio):
    fig, ax1 = plt.subplots()

    # Ensure the index is in datetime format
    portfolio.index = pd.to_datetime(portfolio.index, unit='ms')

    ax1.plot(portfolio.index, portfolio['total'], label='Portfolio Value', color='blue')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Portfolio Value', color='blue')
    ax1.tick_params('y', colors='blue')
    ax1.legend(loc='upper left')

    # Set major ticks format
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.title('Backtest Results: Portfolio Value Over Time')
    plt.xticks(rotation=45)  # Rotate tick marks for better legibility if needed
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels if applied
    plt.show()
