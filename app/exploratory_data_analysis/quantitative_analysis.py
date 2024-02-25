import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.stats import describe, skew, kurtosis, shapiro
import scipy.stats as stats
from scipy.stats import probplot
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.offline import plot


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

    def plot_volatility_clustering(self, price, tilte, num_clusters=5, lambda_value=0.94):
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

        fig.update_layout(title='Volatility Clustering Analysis' + tilte,
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

    def log_returns_statistics(self, log_returns):
        """
        Generate descriptive statistics for log returns.

        Returns:
            tuple: A tuple containing (mean, variance, skewness, kurtosis) of log returns.
        """
        # log_returns = self.calculate_log_returns()
        stats = describe(log_returns)
        skewness = skew(log_returns)
        kurt = kurtosis(log_returns)
        return stats.mean, stats.variance, skewness, kurt

    def volatility_statistics(self, volatility):
        """
        Generate descriptive statistics for volatility.

        Args:
            lambda_value (float): The decay factor for EWMA. Default is 0.94.

        Returns:
            tuple: A tuple containing (mean, variance, skewness, kurtosis) of volatility.
        """
        # volatility = self.estimate_volatility_ewma(lambda_value)
        stats = describe(volatility)
        skewness = skew(volatility)
        kurt = kurtosis(volatility)
        return stats.mean, stats.variance, skewness, kurt

    def print_statistics_results(self, log_returns, volatility):
        log_returns_mean, log_returns_variance, log_returns_skewness, log_returns_kurtosis = self.log_returns_statistics(log_returns)
        print("Log Returns Statistics:")
        print("Mean:", log_returns_mean)
        print("Variance:", log_returns_variance)
        print("Skewness:", log_returns_skewness)
        print("Kurtosis:", log_returns_kurtosis)
        # print(f"Mean: {log_returns_mean},  Variance: {log_returns_variance},  Skewness: {log_returns_skewness},  Kurtosis: {log_returns_kurtosis}")

        volatility_mean, volatility_variance, volatility_skewness, volatility_kurtosis = self.volatility_statistics(volatility)
        print("\nVolatility Statistics:")
        print("Mean:", volatility_mean)
        print("Variance:", volatility_variance)
        print("Skewness:", volatility_skewness)
        print("Kurtosis:", volatility_kurtosis)
        # print(f"Mean: {volatility_mean},  Variance: {volatility_variance},  Skewness: {volatility_skewness},  Kurtosis: {volatility_kurtosis}")



    def plot_qq_plot(self, log_returns, title=""):
        """
        Generate Quantile-Quantile plot to visually inspect the distribution of log returns
        against a normal distribution using Plotly.

        Parameters:
            log_returns (array-like): Array-like object containing log returns data.

        Returns:
            None
        """
        # Generate normal distribution
        mu, sigma = np.mean(log_returns), np.std(log_returns)
        normal_data = np.random.normal(mu, sigma, len(log_returns))

        # Sort the data for plotting
        sorted_log_returns = np.sort(log_returns)
        sorted_normal_data = np.sort(normal_data)

        # Create traces for QQ plot
        qq_trace = go.Scatter(
            x=sorted_normal_data,
            y=sorted_log_returns,
            mode='markers',
            name='Sample Quantiles'
        )

        diagonal_line = go.Scatter(
            x=[np.min(sorted_normal_data), np.max(sorted_normal_data)],
            y=[np.min(sorted_normal_data), np.max(sorted_normal_data)],
            mode='lines',
            name='Diagonal Line',
            line=dict(color='red', dash='dash')
        )

        # Create layout
        layout = go.Layout(
            title='Q-Q Plot of Log Returns Against Normal Distribution' + title,
            xaxis=dict(title='Theoretical Quantiles'),
            yaxis=dict(title='Sample Quantiles'),
            showlegend=True,
            legend=dict(x=0.7, y=0.1)
        )

        # Create figure
        fig = go.Figure(data=[qq_trace, diagonal_line], layout=layout)

        fig.show()
        # plot(fig, filename='qq_plot.html')

    def shapiro_test_log_returns(self, log_returns, alpha=0.05):
        """
        Conduct the Shapiro-Wilk test to quantify the normality of log returns.

        Returns:
            float: The test statistic.
            float: The p-value.
        """
        statistic, p_value = shapiro(log_returns)
        if p_value < alpha:
            print("The data is not normally distributed (reject null hypothesis)")
        else:
            print("There is not enough evidence to conclude that the data is not normally distributed (fail to reject null hypothesis)")

        return statistic, p_value


if __name__ == "__main__":
    import os
    from app.configs import ROOT_DIR
    from app.preprocess import PreProcess
    obj_process = PreProcess()
    obj_analysis = QuantitativeAnalysis()
    df_usd_wallex = pd.read_pickle(os.path.join(ROOT_DIR, "data/USDTTMN_1_2022-12-01_to_2023-12-01_wallex.pkl"))
    df_p = obj_process.construct_price_series(df_usd_wallex, "typical")

    df5 = obj_process.resample_time_scales(df_usd_wallex, "5min", method="VWAP")
    # obj.plot_volatility_clustering(df_p)

    log_return = obj_analysis.calculate_log_returns(df_p)
    volatility = obj_analysis.estimate_volatility_ewma(df_p)

    obj_analysis.print_statistics_results(log_return, volatility)
    # obj_analysis.plot_qq_plot(log_return)

    statistic, p_value = obj_analysis.shapiro_test_log_returns(log_return)

    # random_indices = np.random.choice(log_return.shape[0], 524000, replace=False)

    print("")