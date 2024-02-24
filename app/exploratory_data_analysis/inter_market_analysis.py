import pandas as pd
import numpy as np
from scipy.stats import pearsonr


class InterMarketAnalysis:
    def __init__(self, data):
        self.data = data

    def calculate_log_returns(self):
        return np.log(self.data / self.data.shift(1))

    def synchronous_correlations(self, window_size=30):
        log_returns = self.calculate_log_returns()
        synchronous_corr = log_returns.rolling(window=window_size).corr(pairwise=True)
        return synchronous_corr.xs('USD_TMN', level=1).dropna()

    def lagged_correlations(self, lag=1, window_size=30):
        log_returns = self.calculate_log_returns()
        lagged_returns = log_returns.shift(lag)
        lagged_corr = log_returns.rolling(window=window_size).corr(lagged_returns, pairwise=True)
        return lagged_corr.xs('USD_TMN', level=1).dropna()



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

