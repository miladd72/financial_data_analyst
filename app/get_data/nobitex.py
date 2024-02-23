import requests
import pandas as pd
import os

from app.utils import retry
from app.configs import NOBITEX_URL, RETRY_COUNTS, ROOT_DIR


class GetNobitexData:
    def __init__(self):
        pass

    def get_history_data(self, symbol, start_time, end_time, interval=1, save=False):
        """
        Fetches candlestick data for a given symbol and time range from Nobitex public API.

        Parameters:
            symbol (str): Trading pair symbol (e.g., 'BTCTMN').
            start_time (str): Start time year-month-day.
            end_time (str): End time year-month-day.
            interval (int): Candlestick interval (default: 1).
            save (bool): save the results in data directory

        Returns:
            pd.DataFrame: Candlestick data as a pandas DataFrame.
        """
        page = 1
        all_data = []

        start_time_timestamp = int(pd.Timestamp(start_time).timestamp())
        end_time_timestamp = int(pd.Timestamp(end_time).timestamp())
        i = 0
        while True:
            i += 1
            if i%10 == 0:
                print(f"get epoch: {i}")
            params = {
                'symbol': symbol,
                'resolution': interval,
                'from': start_time_timestamp,
                'to': end_time_timestamp,
                'page': page,
            }

            response = self.__send_requests(params, NOBITEX_URL)
            data = response.json()

            if data['s'] == 'ok':
                all_data.extend(zip(data['t'], data['o'], data['h'], data['l'], data['c'], data['v']))
                page += 1
            elif data['s'] == 'no_data':
                break

        df = self.__process_data(all_data)

        if save:
            adr = os.path.join(ROOT_DIR, f"data/{symbol}_{interval}_{start_time}_to_{end_time}_nobitex.pkl")
            df.to_pickle(adr)
        return df

    def __process_data(self, data):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.insert(0, 'date_time', pd.to_datetime(df['timestamp'].astype(int), unit='s'))
        return df.sort_values(by="date_time").reset_index(drop=True)

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
    nobitex_obj = GetNobitexData()
    symbol = 'BTCIRT'
    interval = 1
    start_time = "2022-12-01"
    end_time = "2023-12-01"
    df = nobitex_obj.get_history_data(symbol=symbol, start_time=start_time, end_time=end_time, save=True)

    print("finish")
