import pandas as pd
import numpy as np
import statsmodels.api as sm


class DynamicCointegrationAnalysis:
    def __init__(self, data1, data2, frequency='ME'):
        self.data1 = data1
        self.data2 = data2
        self.frequency = frequency

    def preprocess_data(self):
        # Combine the two datasets into a single DataFrame
        self.df = pd.concat([self.data1, self.data2], axis=1)
        self.df.columns = ['Series1', 'Series2']

        # Handle missing values if any
        self.df.dropna(inplace=True)

    def split_by_frequency(self):
        # Group data by the specified frequency (e.g., monthly)
        self.groups = self.df.resample(self.frequency)

    def analyze_segments(self):
        results = []

        for _, group in self.groups:
            # Perform cointegration test
            eg_test_result = sm.tsa.coint(group['Series1'], group['Series2'])
            cointegration_result = {
                'Cointegration t-statistic': eg_test_result[0],
                'Cointegration p-value': eg_test_result[1]
            }

            # Perform OLS regression to obtain alpha and beta
            X = sm.add_constant(group['Series1'])
            model = sm.OLS(group['Series2'], X)
            results_ols = model.fit()
            cointegration_result['Alpha'] = results_ols.params[0]
            cointegration_result['Beta'] = results_ols.params[1]

            # Perform stationarity test on residuals
            residuals = results_ols.resid
            adf_test_result = sm.tsa.adfuller(residuals)
            cointegration_result['ADF Statistic'] = adf_test_result[0]
            cointegration_result['ADF p-value'] = adf_test_result[1]

            results.append(cointegration_result)

        return pd.DataFrame(results)

    def run_analysis(self):
        # Preprocessing data
        self.preprocess_data()

        # Splitting data by frequency
        self.split_by_frequency()

        # Analyzing segments
        results = self.analyze_segments()

        return results


if __name__ == "__main__":

    df1 = pd.read_pickle("../error_correction_model/df_usd1_60m")
    df2 = pd.read_pickle("../error_correction_model/df_usd2_60m")
    df3 = pd.read_pickle("../error_correction_model/df_usd3_60m")

    dynamic_analysis = DynamicCointegrationAnalysis(df1.set_index("date_time")["VWAP"], df2.set_index("date_time")["VWAP"], frequency='M')
    results = dynamic_analysis.run_analysis()
    print(results)
    print("")
