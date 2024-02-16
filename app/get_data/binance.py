import requests
import pandas as pd
import os

from app.utils import retry
from app.configs import BINANCE_URL, RETRY_COUNTS, ROOT_DIR


class GetBinanceData:
    def __init__(self):
        pass

    def get_history_data(self, symbol, start_time, end_time, interval='1m', limit=1000, save=False):
        """
        Fetches candlestick data for a given symbol and time range from Binance public API.

        Parameters:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            start_time (str): Start time year-month-day.
            end_time (str): End time in year-month-day.
            interval (str): Candlestick interval (default: '1m').
            limit (int): Number of data points to retrieve per request (maximum is 1000).
            save (bool): save the results in data directory

        Returns:
            pd.DataFrame: Candlestick data as a pandas DataFrame.
        """

        data = []

        start_time_timestamp = int(pd.Timestamp(start_time).timestamp() * 1000)
        end_time_timestamp = int(pd.Timestamp(end_time).timestamp() * 1000)
        i = 0
        while True:
            i += 1
            print(f"get epoch: {i}")
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time_timestamp,
                'endTime': end_time_timestamp,
                'limit': min(limit, 1000),
            }

            response = self.__send_requests(params, BINANCE_URL)
            candlesticks = response.json()

            data.extend(candlesticks)

            if len(candlesticks) < limit:
                break

            start_time_timestamp = int(candlesticks[-1][0]) + 1

        df = self.__process_data(data)

        if save:
            adr = os.path.join(ROOT_DIR, f"data/{symbol}_{interval}_{start_time}_to_{end_time}_binance.pkl")
            df.to_pickle(adr)
        return df

    def __process_data(self, data):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                         'taker_buy_quote_asset_volume', 'ignore'])
        df.insert(0, 'date_time', pd.to_datetime(df['timestamp'], unit='ms'))
        return df.reset_index(drop=True)

    @retry(retry_count=RETRY_COUNTS)
    def __send_requests(self, params, url):
        response = requests.get(url, params=params)
        return self.__response_handler(response)

    def __response_handler(self, response):
        if response.status_code == 200:
            return response

        # here can handle exceptions
        else:
            raise ValueError(f"status code error: {response.status_code}")


if __name__ == "__main__":
    binance_obj = GetBinanceData()
    symbol = 'BTCUSDT'
    interval = '1m'
    limit = 1000
    start_time = "2022-12-01"
    end_time = "2023-12-01"
    df = binance_obj.get_history_data(symbol=symbol, start_time=start_time, end_time=end_time, save=True)

    print("finish")
