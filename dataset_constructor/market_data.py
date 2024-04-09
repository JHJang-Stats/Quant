import pandas as pd

class MarketData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')  # Convert timestamp from ms to datetime
        data = data.set_index('timestamp')  # Set the 'timestamp' column as the index
        return data