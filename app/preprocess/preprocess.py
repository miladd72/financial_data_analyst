import os

import pandas as pd
import numpy as np


class PreProcess:
    def __init__(self):
        pass

    import pandas as pd

    def construct_price_series(self, ohlc_data, price_type='close'):
        """
        Construct a representative price series from OHLCV data.

        Parameters:
            ohlc_data (DataFrame): OHLCV data with columns ['timestamp', 'date_time', 'open', 'high', 'low', 'close', 'volume'].
            price_type (str): Type of price series to construct. Options: 'open', 'high', 'low', 'close',
                              'average', 'typical', 'median', 'weighted_close'.

        Returns:
            Series: Representative price series.
        """

        ohlc_data = ohlc_data.sort_values(by="date_time").set_index("date_time")

        if price_type == 'open':
            price_series = ohlc_data['open']
        elif price_type == 'high':
            price_series = ohlc_data['high']
        elif price_type == 'low':
            price_series = ohlc_data['low']
        elif price_type == 'average':
            price_series = (ohlc_data['open'] + ohlc_data['high'] + ohlc_data['low'] + ohlc_data['close']) / 4
        elif price_type == 'typical':
            price_series = (ohlc_data['high'] + ohlc_data['low'] + ohlc_data['close']) / 3
        elif price_type == 'median':
            price_series = (ohlc_data['high'] + ohlc_data['low']) / 2
        elif price_type == 'weighted_close':
            price_series = (ohlc_data['high'] + ohlc_data['low'] + 2 * ohlc_data['close']) / 4
        else:  # Default to 'close' price
            price_series = ohlc_data['close']

        return price_series

    def calculate_implied_usdt_tmn(self, btc_tmn_prices, btc_usdt_prices):
        """
        Calculate the implied USDT-TMN exchange rate from BTC-TMN and BTC-USDT prices.

        Parameters:
            btc_tmn_prices (Series): BTC-TMN prices.
            btc_usdt_prices (Series): BTC-USDT prices.

        Returns:
            Series: Implied USDT-TMN exchange rate.
        """
        implied_usdt_tmn_rate = btc_tmn_prices / btc_usdt_prices

        return implied_usdt_tmn_rate

    def calculate_usdtmn(self, BTCTMN, BTCUSDT):
        """
        Calculate USDTTMN values from BTCTMN and BTCUSDT OHLCV data.

        Parameters:
            BTCTMN (DataFrame): DataFrame containing OHLCV data for BTCTMN with columns ['open', 'high', 'low', 'close', 'volume', 'date_time'].
            BTCUSDT (DataFrame): DataFrame containing OHLCV data for BTCUSDT with columns ['open', 'high', 'low', 'close', 'volume', 'date_time'].

        Returns:
            DataFrame: DataFrame containing USDTTMN OHLCV data with columns ['date_time', 'open', 'high', 'low', 'close', 'volume'].
        """
        merged_data = pd.merge(BTCTMN, BTCUSDT, on='date_time', suffixes=('_TMN', '_USDT'))

        merged_data['open'] = merged_data['open_TMN'] / merged_data['open_USDT']
        merged_data['high'] = merged_data['high_TMN'] / merged_data['high_USDT']
        merged_data['low'] = merged_data['low_TMN'] / merged_data['low_USDT']
        merged_data['close'] = merged_data['close_TMN'] / merged_data['close_USDT']
        merged_data['volume'] = merged_data['volume_TMN'] / merged_data['close_USDT']

        # Select required columns
        usdttmn_data = merged_data[
            ['date_time', 'open', 'high', 'low', 'close', 'volume']]

        return usdttmn_data

    def resample_time_scales(self, df, time_frame, method='last'):
        """
        Resample minutely data into the specified time frame using the specified resampling method.

        Parameters:
            df (DataFrame): DataFrame containing minutely data with 'date_time' column.
            time_frame (str): Time frame for resampling, e.g., '5T', '20T', '60T', '1440T'.
            method (str): Resampling method, one of 'last', 'TWAP', 'VWAP'. Default is 'last'.

        Returns:
            DataFrame: Resampled DataFrame for the specified time scale with 'date_time' column and additional column based on the resampling method.
        """

        df = df[["date_time", "open", "high", "low", "close", "volume"]]
        df = df.set_index('date_time')

        # Calculate the start time for the first interval
        start_time = df.index.min().floor(time_frame)

        # Resample into specified time frame with the calculated start time
        resampled_df = df.resample(time_frame, origin=start_time).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Additional column based on the resampling method
        if method == 'TWAP':
            # Calculate TWAP
            resampled_df['TWAP'] = df['close'].resample(time_frame, origin=start_time).mean()
        elif method == 'VWAP':
            # Calculate VWAP
            resampled_df['VWAP'] = (df['close'] * df['volume']).resample(time_frame, origin=start_time).sum() / df[
                'volume'].resample(time_frame, origin=start_time).sum()
        elif method == 'last':
            # Use the last recorded price within each time frame
            resampled_df['last'] = df['close'].resample(time_frame, origin=start_time).last()

        # Reset index to convert 'date_time' back into a column
        resampled_df = resampled_df.reset_index()

        return resampled_df





if __name__ == "__main__":
    obj = PreProcess()
    from app.configs import ROOT_DIR
    import os

    # df = pd.read_pickle(os.path.join(ROOT_DIR, "data/btc.pkl"))
    # df_btctmn = pd.read_pickle(os.path.join(ROOT_DIR, "data/BTCTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    # df_btcusdt = pd.read_pickle(os.path.join(ROOT_DIR, "data/BTCUSDT_1m_2022-12-01_to_2023-12-01_binance.pkl"))
    #
    # df_p1 = obj.construct_price_series(df_btctmn, "typical")
    # df_p2 = obj.construct_price_series(df_btcusdt, "typical")
    #
    # df_usdtmn = obj.calculate_usdtmn(df_btctmn, df_btcusdt)
    # df_p_usdtmn = obj.calculate_implied_usdt_tmn(df_p1, df_p2)
    #
    # df_usd_nobitex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTIRT_1_2022-12-01_to_2023-12-01_nobitex.pkl"))
    df_usd_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_p = obj.construct_price_series(df_usd_wallex, "typical")

    df1 = obj.resample_time_scales(df_usd_wallex, "5min", method="VWAP")

    print("")

