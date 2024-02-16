import requests
import pandas as pd
import os

from app.utils import retry
from app.configs import KUCOIN_URL, RETRY_COUNTS, ROOT_DIR


class GetKuCoinData:
    def __init__(self):
        pass

    def get_history_data(self, symbol, start_time, end_time, interval='1min', save=False):
        """
        Fetches candlestick data for a given symbol and time range from KuCoin public API.

        Parameters:
            symbol (str): Trading pair symbol (e.g., 'BTC-USDT').
            start_time (str): Start time year-month-day.
            end_time (str): End time year-month-day.
            interval (str): Candlestick interval (default: '1min').
            save (bool): save the results in data directory

        Returns:
            pd.DataFrame: Candlestick data as a pandas DataFrame.
        """

        data = []

        start_time_timestamp = int(pd.Timestamp(start_time).timestamp())
        end_time_timestamp = int(pd.Timestamp(end_time).timestamp())
        i = 0
        while True:
            i += 1
            print(f"get epoch: {i}")
            params = {
                'symbol': symbol,
                'type': interval,
                'startAt': start_time_timestamp,
                'endAt': end_time_timestamp
            }

            response = self.__send_requests(params, KUCOIN_URL)
            candlesticks = response.json()['data']

            data.extend(candlesticks)

            if len(candlesticks) == 0 :
                break

            end_time_timestamp = int(candlesticks[-1][0]) - 1

        df = self.__process_data(data)

        if save:
            adr = os.path.join(ROOT_DIR, f"data/{symbol}_{interval}_{start_time}_to_{end_time}_kucoin.pkl")
            df.to_pickle(adr)
        return df

    def __process_data(self, data):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'value'])
        df.insert(0, 'date_time', pd.to_datetime(df['timestamp'].astype(int), unit='s'))
        return df

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
    kucoin_obj = GetKuCoinData()
    symbol = 'BTC-USDT'  # KuCoin symbol format
    interval = '1min'    # KuCoin uses different interval format
    start_time = "2022-12-01"
    end_time = "2023-12-01"
    df = kucoin_obj.get_history_data(symbol=symbol, start_time=start_time, end_time=end_time, save=True)

    print("finish")
