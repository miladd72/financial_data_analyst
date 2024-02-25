import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objs as go


class InterMarketAnalysis:

    def __init__(self, data):
        """
        Initialize InterMarketAnalysis object with the provided data.

        Parameters:
        data (DataFrame): The DataFrame containing USDT-TMN prices data.
        """
        self.data = data

    def calculate_log_returns(self):
        """
        Calculate the log returns of the USDT-TMN prices.

        Returns:
        DataFrame: A DataFrame containing log returns of the USDT-TMN prices.
        """
        return np.log(self.data / self.data.shift(1))

    def get_window_size(self, freq):
        """
        Calculate the appropriate window size for the rolling window based on data frequency.

        Parameters:
        freq (str): The frequency of the data (e.g., '1min', '5min', '20min', '60min', '1440min').

        Returns:
        int: The window size for the rolling window.
        """
        if freq.endswith('min'):
            minutes = int(freq[:-3])
            return (1440 // minutes) * 30  # 30 days equivalent
        elif freq == 'h':
            return 30 * 24
        elif freq == 'D':
            return 30
        else:
            raise ValueError("Unsupported frequency")

    def synchronous_correlations(self):
        """
        Calculate synchronous correlations between different USDT-TMN prices.

        Returns:
        DataFrame: A DataFrame containing synchronous correlations between different USDT-TMN prices.
        """
        log_returns = self.calculate_log_returns()
        window_size = self.get_window_size(log_returns.index.freqstr)
        synchronous_corr = log_returns.rolling(window=window_size).corr(pairwise=True)
        result = self.extract_pairwise_corr(synchronous_corr.dropna())
        return result

    def lagged_correlations(self, lag=1):
        """
        Calculate lagged correlations between different USDT-TMN prices.

        Parameters:
        lag (int): The lag for calculating correlations.

        Returns:
        DataFrame: A DataFrame containing lagged correlations between different USDT-TMN prices.
        """
        log_returns = self.calculate_log_returns()
        window_size = self.get_window_size(log_returns.index.freqstr)
        lagged_returns = log_returns.shift(lag)
        lagged_corr = log_returns.rolling(window=window_size).corr(lagged_returns, pairwise=True)
        result = self.extract_pairwise_corr(lagged_corr.dropna())
        return result

    def extract_pairwise_corr(self, df_corr):
        """
        Extract pairwise correlations between different USDT-TMN prices.

        Parameters:
        df_corr (DataFrame): DataFrame containing correlations.

        Returns:
        DataFrame: A DataFrame containing pairwise correlations between different USDT-TMN prices.
        """
        columns = df_corr.columns
        pairwise_correlations = []

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                pair_corr = df_corr.xs(col1, level=1)[col2].rename(f"{col1}_{col2}")
                pairwise_correlations.append(pair_corr)

        return pd.concat(pairwise_correlations, axis=1)

    def plot_correlation(self, series_data, title):
        """
        Plot a series data using Plotly.

        Parameters:
            series_data (pandas.Series): Series data to be plotted.
            title (str): Title of the plot.

        Returns:
            None (displays the plot).
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=series_data.index, y=series_data.values, mode='lines+markers', name='Data'))

        fig.update_layout(title=title,
                          xaxis_title='Datetime',
                          yaxis_title='Value',
                          template='plotly_dark')

        # Show the plot
        fig.show()




if __name__ == '__main__':
    import os
    from app.configs import ROOT_DIR
    from app.preprocess import PreProcess
    from app.exploratory_data_analysis import QuantitativeAnalysis

    obj_process = PreProcess()
    obj_analysis = QuantitativeAnalysis()
    df_usd_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_p = obj_process.construct_price_series(df_usd_wallex, "typical")

    df5 = obj_process.resample_time_scales(df_usd_wallex, "5min", method="VWAP")
    # obj.plot_volatility_clustering(df_p)

    log_return = obj_analysis.calculate_log_returns(df_p)

