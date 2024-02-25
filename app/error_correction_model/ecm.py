import numpy as np
import pandas as pd

class ErrorCorrectionModel:

    def __init__(self, data_x, data_y):
        """
        Initialize the ErrorCorrectionModel object.

        Parameters:
        data_x (list or array): The first time series data.
        data_y (list or array): The second time series data.
        """
        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        self.residuals = None
        self.lagged_data = None
        self.ecm_coefficients = None
        self.durbin_watson_statistic = None

    def estimate_cointegration(self):
        """
        Estimate the cointegration relationship using the Engle-Granger method.
        """
        # Step 1: Estimate the regression coefficients
        ones = np.ones(len(self.data_x))
        X = np.column_stack((ones, self.data_x))
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ self.data_y

        # Step 2: Calculate the residuals
        self.residuals = self.data_y - X @ beta_hat

    def determine_lag_order(self, lag_order):
        """
        Determine the lag order for the Error Correction Model.

        Parameters:
        lag_order (int): The lag order to be used for the ECM.
        """
        self.lag_order = lag_order

    def estimate_ecm(self):
        """
        Estimate the parameters of the Error Correction Model using Ordinary Least Squares (OLS).
        """
        # Step 1: Construct lagged variables
        lagged_data_x = np.column_stack((np.ones(len(self.data_x) - 1), self.data_x[:-1]))
        lagged_data_y = self.data_y[:-1]

        # Step 2: Construct lagged dependent variable
        lagged_y = self.data_y[1:]

        # Store lagged data
        self.lagged_data = np.column_stack((lagged_data_x, lagged_y))

        # Step 3: Estimate ECM parameters using OLS
        self.ecm_coefficients = np.linalg.inv(self.lagged_data.T @ self.lagged_data) @ self.lagged_data.T @ self.residuals[:-1]  # Adjusted dimensions

    def analyze_dynamics(self):
        """
        Analyze the reversion dynamics of the Error Correction Model.
        """
        # Calculate residuals of the ECM
        fitted_values = self.lagged_data @ self.ecm_coefficients
        residuals_ecm = self.residuals[:-1] - fitted_values

        # Calculate Durbin-Watson statistic
        num = np.sum(np.diff(residuals_ecm) ** 2)
        den = np.sum(residuals_ecm ** 2)
        self.durbin_watson_statistic = num / den

    def summary(self):
        """
        Output a summary of the ECM analysis results.
        """
        print("ECM Coefficients:")
        print("Alpha (lagged dependent variable coefficient):", self.ecm_coefficients[1])
        print("Beta (error correction term coefficient):", self.ecm_coefficients[2])
        print("\nDiagnostic Tests:")
        print("Durbin-Watson statistic:", self.durbin_watson_statistic)




if __name__ == "__main__":
    df1 = pd.read_pickle("df_usd1_60m")
    df2 = pd.read_pickle("df_usd2_60m")
    df3 = pd.read_pickle("df_usd3_60m")
    data_x = df2.set_index("date_time")["VWAP"]
    data_y = df3.set_index("date_time")["VWAP"]

    # Create an instance of ErrorCorrectionModel
    ecm = ErrorCorrectionModel(data_x, data_y)

    # Perform ECM analysis
    ecm.estimate_cointegration()
    ecm.determine_lag_order(1)  # Example lag order
    ecm.estimate_ecm()
    ecm.analyze_dynamics()

    # Output summary
    ecm.summary()
