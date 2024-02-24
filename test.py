from datetime import datetime

datetime.now()
#%%

# Create a datetime object for December 1, 2022
dt = datetime(2022, 12, 1)

# Convert the datetime object to seconds
seconds = dt.timestamp()
print(seconds)


dt = datetime(2023, 12, 1)

# Convert the datetime object to seconds
seconds = dt.timestamp()
print(seconds)
#%%
from kucoin.client import Client
import datetime

# Initialize the KuCoin client
client = Client()

# Define the start and end timestamps
start_timestamp = datetime.datetime(2022, 12, 1).timestamp() * 1000  # Convert to milliseconds
end_timestamp = datetime.datetime(2022, 12, 2).timestamp() * 1000  # Convert to milliseconds

# Get 1-minute historical data for BTCUSDT
historical_data = client.get_kline_data('BTC-USDT', '1min', start=start_timestamp, end=end_timestamp)

# Print the historical data
for data_point in historical_data:
    print(data_point)

#%%
import requests
import datetime
import pandas as pd

# Define the API endpoint
endpoint = 'https://api.kucoin.com/api/v1/market/candles'

# Define the parameters
symbol = 'BTC-USDT'
start_timestamp = int(datetime.datetime(2024, 2, 1).timestamp())
end_timestamp = int(datetime.datetime(2024, 2, 2).timestamp())
interval = '1min'

# Make the request
response = requests.get(endpoint, params={'symbol': symbol, 'startAt': start_timestamp, 'endAt': end_timestamp, 'type': interval})

# Check if request was successful
if response.status_code == 200:
    historical_data = response.json()

    df = pd.DataFrame(historical_data['data'], columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # # Print or process the historical data
    # for data_point in historical_data:
    #     print(data_point)
else:
    print(f"Error: {response.status_code} - {response.text}")

#%%
import pandas as pd
df = pd.DataFrame(historical_data['data'], columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
    # Convert time to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')


#%%

import requests
import pandas as pd
import datetime


def fetch_binance_historical_data(symbol, start_time, end_time, interval='1m'):
    endpoint = 'https://api.binance.com/api/v3/klines'
    limit = 1000  # Maximum number of data points per request

    all_data = []
    start_str = str(int(start_time.timestamp() * 1000))
    end_str = str(int(end_time.timestamp() * 1000))

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_str,
        'endTime': end_str,
        'limit': limit
    }

    while True:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break  # No more data available
            all_data.extend(data)
            # Update start time for next request
            start_str = str(int(data[-1][0]))
            params['startTime'] = start_str
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break

    return all_data


symbol = 'BTCUSDT'
start_time = datetime.datetime(2022, 12, 1)
end_time = datetime.datetime(2023, 12, 1)
interval = '1m'

historical_data = fetch_binance_historical_data(symbol, start_time, end_time, interval)

# Convert data to DataFrame
df = pd.DataFrame(historical_data,
                  columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                           'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(df.head())


#%%
import requests
import pandas as pd

# Binance API endpoint for klines (candlestick data)
url = 'https://api.binance.com/api/v3/klines'


start_time = int(pd.Timestamp('2023-12-01 00:00:00').timestamp() * 1000)
end_time = int(pd.Timestamp('2023-12-02 00:00:00').timestamp() * 1000)

# Parameters for the request
params = {
    'symbol': 'BTCUSDT',  # Trading pair symbol
    'interval': '1m',      # Interval (1 minute)
    'startTime': start_time,
    'endTime': end_time,
    'limit': 2000          # Number of data points to retrieve (maximum is 1000)
}

# Send GET request to Binance API
response = requests.get(url, params=params)

# Convert the JSON response to a DataFrame
data = pd.DataFrame(response.json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert timestamp to readable date format
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

# Set timestamp as index
data.set_index('timestamp', inplace=True)

print(data)

#%%
import requests
import pandas as pd

def fetch_candlestick_data(symbol, start_time, end_time, interval='1m', limit=1000):
    """
    Fetches candlestick data for a given symbol and time range from Binance public API.

    Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        start_time (int): Start time in milliseconds since epoch.
        end_time (int): End time in milliseconds since epoch.
        interval (str): Candlestick interval (default: '1m').
        limit (int): Number of data points to retrieve per request (maximum is 1000).

    Returns:
        pd.DataFrame: Candlestick data as a pandas DataFrame.
    """
    url = 'https://api.binance.com/api/v3/klines'
    data = []

    # Convert start and end time to ISO format
    start_time_iso = pd.to_datetime(start_time, unit='ms').isoformat()
    end_time_iso = pd.to_datetime(end_time, unit='ms').isoformat()

    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': min(limit, 2000),  # Ensure limit does not exceed 1000
        }

        response = requests.get(url, params=params)
        candlesticks = response.json()

        # Append data to list
        data.extend(candlesticks)

        # If the number of retrieved candlesticks is less than the limit, break the loop
        if len(candlesticks) < limit:
            break

        # Update start_time for the next request
        start_time = int(candlesticks[-1][0]) + 1  # Set start_time to the timestamp of the last candlestick + 1

    # Convert the list of candlesticks to a DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    # Convert timestamp to readable date format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    return df

# Example usage:
start_time = pd.Timestamp('2024-01-01 00:00:00').timestamp() * 1000  # Convert to milliseconds
end_time = pd.Timestamp('2024-01-02 00:00:00').timestamp() * 1000  # Convert to milliseconds

symbol = 'BTCUSDT'
interval = '1m'
limit = 500  # Example limit

data = fetch_candlestick_data(symbol, start_time, end_time, interval, limit)
print(data)

#%%


import pandas as pd
import numpy as np

import pandas as pd

import pandas as pd

def fill_missing_data(df, method='carry_forward'):
    """
    Fill missing data in the DataFrame using the specified method.

    Parameters:
        df (DataFrame): DataFrame containing financial time series data.
        method (str): Method for filling missing data. Options are 'carry_forward', 'simple_average', or 'adjacent_mean'.
                      Default is 'carry_forward'.

    Returns:
        DataFrame: DataFrame with missing values filled according to the specified method.
    """
    if method == 'carry_forward':
        # Fill missing values with the last known value (carry forward)
        filled_df = df.fillna(method='ffill')
    elif method == 'simple_average':
        # Fill missing values with the simple average of adjacent data points for each column
        filled_df = df.apply(lambda col: col.fillna(col.interpolate()), axis=0)
    elif method == 'adjacent_mean':
        # Fill missing values with the mean of 10 adjacent values (5 before and 5 after)
        filled_df = df.copy()
        for col in filled_df.columns:
            missing_indices = filled_df[col][filled_df[col].isnull()].index
            for idx in missing_indices:
                start_idx = max(0, idx - 5)
                end_idx = min(len(filled_df) - 1, idx + 5)
                adjacent_values = filled_df[col].iloc[start_idx:end_idx+1]
                filled_df.at[idx, col] = adjacent_values.mean()
    else:
        raise ValueError("Invalid method. Supported methods are 'carry_forward', 'simple_average', and 'adjacent_mean'.")

    return filled_df

# Example usage:
# df is a DataFrame containing financial time series data
# filled_df = fill_missing_data(df, method='adjacent_mean')


# Example usage:
# df is a DataFrame containing financial time series data
# filled_df = fill_missing_data(df, method='carry_forward'


# Create a sample OHLCV DataFrame with missing values
data = {
    'date_time': pd.date_range(start='2024-01-05 01:00:00', end='2024-01-05 17:00:00', freq='h'),
    'open': [50, 105, np.nan, 110, 115, np.nan, 120, 125, 130, 135, 140, np.nan, 145, 150, 155, np.nan, 160],
    'high': [105, 110, np.nan, 115, 120, np.nan, 125, 130, 135, 140, 145, np.nan, 150, 155, 160, np.nan, 165],
    'low': [95, 100, np.nan, 105, 110, np.nan, 115, 120, 125, 130, 135, np.nan, 140, 145, 150, np.nan, 155],
    'close': [102, 108, np.nan, 112, 118, np.nan, 122, 128, 132, 138, 142, np.nan, 148, 152, 158, np.nan, 162],
    'volume': [1000, 1100, np.nan, 1200, 1300, np.nan, 1400, 1500, 1600, 1700, 1800, np.nan, 1900, 2000, 2100, np.nan, 2200]
}

df = pd.DataFrame(data)

# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Fill missing values using carry forward method
filled_df = fill_missing_data(df, method='adjacent_mean')

# Print the DataFrame after filling missing values
print("\nDataFrame after filling missing values:")
print(filled_df)

