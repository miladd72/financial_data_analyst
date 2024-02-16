import aiohttp
import asyncio
import pandas as pd
import os

from app.utils import retry
from app.configs import BINANCE_URL, RETRY_COUNTS, ROOT_DIR


class GetBinanceData:
    def __init__(self):
        pass

    async def fetch_data(self, session, symbol, interval, start_time, end_time, limit):
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000),
        }
        start_time_timestamp = int(pd.Timestamp(start_time).timestamp() * 1000)
        end_time_timestamp = int(pd.Timestamp(end_time).timestamp() * 1000)

        while start_time_timestamp < end_time_timestamp:
            params['startTime'] = start_time_timestamp
            params['endTime'] = min(start_time_timestamp + limit * 60000, end_time_timestamp)

            async with session.get(BINANCE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    yield data
                else:
                    raise ValueError(f"status code error: {response.status}")

            start_time_timestamp = params['endTime'] + 60000

    async def get_history_data(self, symbol, start_time, end_time, interval='1m', limit=1000, save=False):
        """
        Fetches candlestick data for a given symbol and time range from Binance public API asynchronously.

        Parameters:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            start_time (str): Start time in milliseconds since epoch.
            end_time (str): End time in milliseconds since epoch.
            interval (str): Candlestick interval (default: '1m').
            limit (int): Number of data points to retrieve per request (maximum is 1000).
            save (bool): save the results in data directory

        Returns:
            pd.DataFrame: Candlestick data as a pandas DataFrame.
        """

        async with aiohttp.ClientSession() as session:
            tasks = []
            async for data in self.fetch_data(session, symbol, interval, start_time, end_time, limit):
                tasks.append(data)

            responses = await asyncio.gather(*tasks)

        df = self.__process_data([item for sublist in responses for item in sublist])

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


if __name__ == "__main__":
    async def main():
        binance_obj = GetBinanceData()
        symbol = 'BTCUSDT'
        interval = '1m'
        limit = 1000
        start_time = "2024-01-01"
        end_time = "2024-02-10"
        df = await binance_obj.get_history_data(symbol=symbol, start_time=start_time, end_time=end_time, save=True)
        print("finish")

    asyncio.run(main())
