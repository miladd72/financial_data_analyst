import pandas as pd
import os
from app.configs import ROOT_DIR
from app.preprocess import PreProcess
from app.exploratory_data_analysis import QuantitativeAnalysis, InterMarketAnalysis

if __name__ == '__main__':


    obj_process = PreProcess()
    obj_analysis = QuantitativeAnalysis()


    df_usd_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_usd_nobitex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTIRT_1_2022-12-01_to_2023-12-01_nobitex.pkl"))
    df_btctmn_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/BTCTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_btcusd_binance = pd.read_pickle(os.path.join(ROOT_DIR, "data/BTCUSDT_1m_2022-12-01_to_2023-12-01_binance.pkl"))

    df_usd = obj_process.calculate_usdtmn(df_btctmn_wallex, df_btcusd_binance)

    df1 = obj_process.resample_time_scales(df_usd_wallex, "20min", method="VWAP")
    df2 = obj_process.resample_time_scales(df_usd_nobitex, "20min", method="VWAP")
    df3 = obj_process.resample_time_scales(df_usd, "20min", method="VWAP")

    df1_p = obj_process.construct_price_series(df1, "typical")
    df2_p = obj_process.construct_price_series(df2, "typical")
    df3_p = obj_process.construct_price_series(df3, "typical")

    data = pd.concat([df1_p, df2_p, df3_p], axis=1)
    data.columns = ['USD_TMN1', 'USD_TMN2', 'USD_TMN3']

    obj_inter_market = InterMarketAnalysis(data)
    synchronous_corr = obj_inter_market.synchronous_correlations()
    print("Synchronous Correlations:")
    # print(synchronous_corr)

    # Calculate lagged correlations
    lagged_corr = obj_inter_market.lagged_correlations()
    print("\nLagged Correlations (Lag 1):")
    # print(lagged_corr)



    # df5 = obj_process.resample_time_scales(df_usd_wallex, "5min", method="VWAP")
    # obj.plot_volatility_clustering(df_p)

    # log_return = obj_analysis.calculate_log_returns(df_p)

    # import plotly.graph_objs as go
    #
    # fig = go.Figure()
    #
    # # Add a scatter plot trace
    # fig.add_trace(go.Scatter(x=synchronous_corr.index, y=synchronous_corr['USD_TMN1_USD_TMN2'], mode='lines+markers', name='Correlation'))
    #
    # # Update layout
    # fig.update_layout(title='Hourly Correlation between Two Prices',
    #                   xaxis_title='Datetime',
    #                   yaxis_title='Correlation',
    #                   template='plotly_dark')
    #
    # # Show the plot
    # fig.show()


    obj_inter_market.plot_correlation(synchronous_corr["USD_TMN1_USD_TMN2"], title="USD_TMN1_USD_TMN2 correlations")
    print("")