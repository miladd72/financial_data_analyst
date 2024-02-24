import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans


class QuantitativeAnalysis:
    def __init__(self):
        pass

    def calculate_log_returns(self, price):
        """
        Calculate the log returns of the provided prices.

        Returns:
            numpy.ndarray: An array containing the calculated log returns.

        Raises:
            ValueError: If less than two prices are provided.
        """
        if len(price) < 2:
            raise ValueError("At least two prices are required to calculate log returns.")

        log_returns = np.diff(np.log(price))
        return log_returns

    def estimate_volatility_ewma(self, price,  lambda_value=0.94):
        """
        Estimate volatility using the Exponentially Weighted Moving Average (EWMA) model.

        Args:
            lambda_value (float): The decay factor for EWMA. Default is 0.94.

        Returns:
            numpy.ndarray: An array containing the estimated volatility.
        """
        log_returns = self.calculate_log_returns(price)
        volatilities = np.zeros_like(log_returns)
        volatilities[0] = np.var(log_returns)

        for t in range(1, len(log_returns)):
            volatilities[t] = lambda_value * volatilities[t - 1] + (1 - lambda_value) * log_returns[t - 1] ** 2

        return np.sqrt(volatilities)

    def cluster_volatility(self, volatility, num_clusters=3):
        """
        Cluster volatility into distinct market regimes using K-means clustering.

        Args:
            num_clusters (int): Number of clusters to create. Default is 3.
            lambda_value (float): The decay factor for EWMA. Default is 0.94.
        """
        # volatility = self.estimate_volatility_ewma(price, lambda_value)

        # Reshape volatility data for clustering
        volatility = volatility.reshape(-1, 1)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        labels = kmeans.fit_predict(volatility)

        return labels

    def plot_volatility_clustering(self, price, num_clusters=5, lambda_value=0.94):
        """
        Visualize volatility clustering to discern distinct market regimes.

        Args:
            num_clusters (int): Number of clusters to create. Default is 3.
            lambda_value (float): The decay factor for EWMA. Default is 0.94.
        """
        volatility = self.estimate_volatility_ewma(price, lambda_value)
        labels = self.cluster_volatility(volatility, num_clusters)

        fig = go.Figure()
        for cluster_label in range(num_clusters):
            cluster_indices = np.where(labels == cluster_label)[0]
            fig.add_trace(go.Scatter(x=cluster_indices, y=volatility[cluster_indices],
                                     mode='lines', name=f'Cluster {cluster_label}'))

        fig.update_layout(title='Volatility Clustering Analysis',
                          xaxis_title='Time',
                          yaxis_title='Volatility',
                          template='plotly_dark')
        fig.show()

    # def plot_volatility_clustering(self, price,  lambda_value=0.94):
    #     """
    #     Visualize volatility clustering using Plotly.
    #
    #     Args:
    #         lambda_value (float): The decay factor for EWMA. Default is 0.94.
    #     """
    #     volatility = self.estimate_volatility_ewma(price, lambda_value)
    #
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=list(range(len(volatility))), y=volatility, mode='lines', name='Volatility'))
    #     fig.update_layout(title='Volatility Clustering Analysis',
    #                       xaxis_title='Time',
    #                       yaxis_title='Volatility',
    #                       template='plotly_dark')
    #     fig.show()


if __name__ == "__main__":
    import os
    from app.configs import ROOT_DIR
    from app.preprocess import PreProcess
    obj_process = PreProcess()
    obj = QuantitativeAnalysis()
    df_usd_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_p = obj_process.construct_price_series(df_usd_wallex, "typical")

    df5 = obj_process.resample_time_scales(df_usd_wallex, "5min", method="VWAP")
    obj.plot_volatility_clustering(df_p)
    print("")