import requests
import pandas as pd
from app.utils import retry


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
    i = 0
    while True:
        i += 1
        print(i)
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': min(limit, 1000),  # Ensure limit does not exceed 1000
        }



        response = send_requests(params, url)
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

@retry(retry_count=3)
def send_requests(params, url):
    response = requests.get(url, params=params)
    return response


# Example usage:
start_time = int(pd.Timestamp('2023-01-01 00:00:00').timestamp() * 1000)  # Convert to milliseconds
end_time = int(pd.Timestamp('2023-12-01 00:00:00').timestamp() * 1000)  # Convert to milliseconds

symbol = 'BTCUSDT'
interval = '1m'
limit = 1000  # Example limit

data = fetch_candlestick_data(symbol, start_time, end_time, interval, limit)
print(data)


#%%
import os
from app.configs import ROOT_DIR
df1 = pd.read_pickle(os.path.join(ROOT_DIR, "data\BTC-USDT_1min_2022-12-01_to_2023-12-01_kucoin.pkl"))
df2 = pd.read_pickle(os.path.join(ROOT_DIR, "data\BTCUSDT_1m_2022-12-01_to_2023-12-01_binance.pkl"))

#%%
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats

# Define parameters
mean = 6.312159199465759e-07
variance = 3.7062057427773346e-07
skewness = -0.5735230695186057
kurtosis = 910.0157718078657


# Mean = 0.0005030122238542142
# Variance = 1.1761032059723276e-07
# Skewness = 11.597519331962822
# Kurtosis = 485.93204975971645


# Generate random data based on the moments
data = stats.skewnorm.rvs(a=skewness, loc=mean, scale=np.sqrt(variance), size=10000)

# Create histogram
hist_data, bin_edges = np.histogram(data, bins=100, density=True)
bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

# Create figure
fig = go.Figure()

# Add histogram trace
fig.add_trace(go.Bar(x=bin_centers, y=hist_data,
                     marker_color='rgba(0, 0, 255, 0.5)',
                     name='Histogram'))

# Update layout
fig.update_layout(title='Distribution Plot',
                  xaxis_title='Value',
                  yaxis_title='Density')

# Show figure
fig.show()
#%%

