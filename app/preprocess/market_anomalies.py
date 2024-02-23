import pandas as pd
import numpy as np



class MarketAnomalies:
    def __init__(self):
        self.price_col = ["open", "high", "low", "close"]

    def fill_missing_data(self, df, method='carry_forward'):
        """
        Fill missing data in the DataFrame using the specified method.

        Parameters:
            df (DataFrame): DataFrame containing financial time series data.
            method (str): Method for filling missing data. Options are 'carry_forward', 'simple_average', or 'adjacent_mean'.
                          Default is 'carry_forward'.

        Returns:
            DataFrame: DataFrame with missing values filled according to the specified method.
        """
        if method == 'carry_forward':
            filled_df = df.fillna(method='ffill')
        elif method == 'simple_average':
            filled_df = df.apply(lambda col: col.fillna(col.interpolate()), axis=0)
        elif method == 'adjacent_mean':
            # Fill missing values with the mean of adjacent values (n before and n after)
            filled_df = self.fill_adjacent_mean(df, n=4)
        else:
            raise ValueError(
                "Invalid method. Supported methods are 'carry_forward', 'simple_average', and 'adjacent_mean'.")

        return filled_df

    def fill_adjacent_mean(self, df, n=4):
        filled_df = df.copy()
        for col in filled_df.columns:
            missing_indices = filled_df[col][filled_df[col].isnull()].index
            for idx in missing_indices:
                start_idx = max(0, idx - n)
                end_idx = min(len(filled_df) - 1, idx + n)
                adjacent_values = filled_df[col].iloc[start_idx:end_idx + 1]
                filled_df.at[idx, col] = adjacent_values.mean()
        return filled_df

    def _detect_zscore_outliers(self, df, threshold=3):
        """
        Detect outliers using z-scores for each column separately.

        Parameters:
            threshold (float): Threshold for z-score. Default is 3.

        Returns:
            DataFrame: Boolean DataFrame indicating outliers.
        """
        z_scores = (df - df.mean()) / df.std()
        return np.abs(z_scores) > threshold

    def _detect_iqr_outliers(self, df, threshold=1.5):
        """
        Detect outliers using interquartile range (IQR) for each column separately.

        Parameters:
            threshold (float): Threshold for IQR multiplier. Default is 1.5.

        Returns:
            DataFrame: Boolean DataFrame indicating outliers.
        """
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        return (df < (q1 - threshold * iqr)) | (df > (q3 + threshold * iqr))

    def _detect_outliers_time_series(self, df, window_size=5, threshold=2):
        """
        Detect outliers in a time series data using a rolling window approach.

        Parameters:
            df (DataFrame): The DataFrame containing the time series data.
            window_size (int): The size of the rolling window. Default is 5.
            threshold (float): The threshold for considering a value as an outlier.
                               Default is 2.

        Returns:
            DataFrame: A boolean DataFrame indicating the outlier indices.
        """

        rolling_mean = df.rolling(window=window_size, min_periods=1).mean()
        rolling_st = df.rolling(window=window_size, min_periods=1).std()

        outlier_indices = (df > (rolling_mean + threshold * rolling_mean)) | \
                          (df < (rolling_st - threshold * rolling_st))

        return outlier_indices

    def detect_and_correct_outliers(self, df, method='z-score', threshold=3, interp_method='linear'):
        """
        Detect and correct outliers in the DataFrame.

        Parameters:
            df (dataframe): OHLCV data price
            method (str): Method for identifying outliers. Options are 'z-score' or 'boxplot' ot 'time_series. Default is 'z-score'.
            threshold (float): Threshold for z-score or IQR multiplier. Default is 3 for z-score and 1.5 for IQR.
            interp_method (str): Method for interpolation if outliers are detected. Options are 'linear', 'quadratic', 'cubic', etc. Default is 'linear'.

        Returns:
            DataFrame: DataFrame with outliers corrected or excluded.
        """

        # Correct outliers based on interpolation
        corrected_df = df.copy()
        for col in self.price_col:
            if method == 'z-score':
                outliers = self._detect_zscore_outliers(corrected_df[col], threshold)
            elif method == 'boxplot':
                outliers = self._detect_iqr_outliers(corrected_df[col], threshold)
            elif method == "time_series":
                outliers = self._detect_outliers_time_series(corrected_df[col], window_size=5, threshold=2)
            else:
                raise ValueError("Invalid method. Supported methods are 'z-score' and 'boxplot'.")

            print(sum(outliers))
            corrected_df[col] = corrected_df[col].mask(outliers, np.nan)
            if interp_method != 'exclude':
                corrected_df[col] = corrected_df[col].mask(outliers, corrected_df[col].interpolate(method=interp_method))


        return corrected_df


if __name__ == "__main__":
    from app.configs import ROOT_DIR
    import os
    obj = MarketAnomalies()
    df = pd.read_pickle(os.path.join(ROOT_DIR, "data/usd.pkl"))
    dfr = obj.detect_and_correct_outliers(df, method='time_series', threshold=3, interp_method='linear')
    print("")