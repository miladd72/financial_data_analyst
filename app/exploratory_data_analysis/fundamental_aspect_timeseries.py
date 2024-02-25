import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from plotly.subplots import make_subplots


class AutocorrelationStationaryAnalysis:
    def __init__(self):
        pass

    def _plot_acf(self, series, title):
        """
                Generate and display ACF plot for the given time series.

                Args:
                series (pd.Series): The time series data.
                title (str): The title for the plot.
        """
        lag_acf = acf(series, nlags=20)
        acf_trace = go.Scatter(x=np.arange(len(lag_acf)), y=lag_acf, mode='lines+markers', name=title)
        layout = go.Layout(title=title, xaxis=dict(title='Lag'), yaxis=dict(title='ACF'))
        fig = go.Figure(data=[acf_trace], layout=layout,)
        # fig.show()
        return fig

    def plot_acf(self, price, log_return, volatility, title):
        fig1 = self._plot_acf(price, "price")
        fig2 = self._plot_acf(log_return, "log_return")
        fig3 = self._plot_acf(volatility, "volatility")

        fig = make_subplots(rows=1, cols=3)

        for i, fig_obj in enumerate([fig1, fig2, fig3], start=1):
            for trace in fig_obj['data']:
                fig.add_trace(trace, row=1, col=i)

            fig.update_xaxes(title_text=fig_obj.layout.xaxis.title.text, row=1, col=i)
            fig.update_yaxes(title_text=fig_obj.layout.yaxis.title.text, row=1, col=i)
            # fig.update_layout(title_text=fig_obj.layout.title.text + f" ({i})", row=1, col=i)

        fig.update_layout(title_text=title)
        fig.show()


    def _plot_pacf(self, series, title):
        """
                Generate and display PACF plot for the given time series.

                Args:
                series (pd.Series): The time series data.
                title (str): The title for the plot.
        """
        lag_pacf = pacf(series, nlags=20)
        pacf_trace = go.Scatter(x=np.arange(len(lag_pacf)), y=lag_pacf, mode='lines+markers')
        layout = go.Layout(title=title, xaxis=dict(title='Lag'), yaxis=dict(title='PACF'))
        fig = go.Figure(data=[pacf_trace], layout=layout)
        # fig.show()
        return fig

    def plot_pacf(self, price, log_return, volatility, title):
        fig1 = self._plot_pacf(price, "price")
        fig2 = self._plot_pacf(log_return, "log_return")
        fig3 = self._plot_pacf(volatility, "volatility")

        fig = make_subplots(rows=1, cols=3)

        for i, fig_obj in enumerate([fig1, fig2, fig3], start=1):
            for trace in fig_obj['data']:
                fig.add_trace(trace, row=1, col=i)

            fig.update_xaxes(title_text=fig_obj.layout.xaxis.title.text, row=1, col=i)
            fig.update_yaxes(title_text=fig_obj.layout.yaxis.title.text, row=1, col=i)
            # fig.update_layout(title_text=fig_obj.layout.title.text + f" ({i})", row=1, col=i)

        fig.update_layout(title_text=title)
        fig.show()

    def adf_test(self, series, title):
        """
        Perform Augmented Dickey-Fuller (ADF) test on the given time series.

        Args:
        series (pd.Series): The time series data.
        title (str): Title for the output.

        Returns:
        tuple: A tuple containing ADF test results.
        """
        result = adfuller(series)
        print(title)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value}')
        print('Conclusion:')
        if result[1] <= 0.01:
            print('Reject the null hypothesis (H0): The series is stationary.')
        else:
            print('Fail to reject the null hypothesis (H0): The series is non-stationary.')
        return result


if __name__ == "__main__":
    import os
    from app.configs import ROOT_DIR
    from app.preprocess import PreProcess
    from app.exploratory_data_analysis import QuantitativeAnalysis

    obj_process = PreProcess()
    obj_analysis = QuantitativeAnalysis()
    obj_ac = AutocorrelationStationaryAnalysis()
    df_usd_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_p = obj_process.construct_price_series(df_usd_wallex, "typical")

    df5 = obj_process.resample_time_scales(df_usd_wallex, "5min", method="VWAP")
    # obj.plot_volatility_clustering(df_p)

    log_return = obj_analysis.calculate_log_returns(df_p)
    volatility = obj_analysis.estimate_volatility_ewma(df_p)

    # obj_analysis.print_statistics_results(log_return, volatility)

    # obj_ac.plot_acf(series=log_return, title="log returns")
    # obj_ac.plot_pacf(series=log_return, title="log returns")
    # obj_ac.adf_test(log_return, title="adf")

    obj_ac.plot_acf(df_p, log_return, volatility)



    print("")