import pandas as pd
import numpy as np
import statsmodels.api as sm

class CointegrationAnalysis:
    """
    A class to perform cointegration analysis between two time series.
    """

    def __init__(self):
        pass

    def cointegration_test(self, price1: pd.Series, price2: pd.Series):
        """
        Perform Engle-Granger cointegration test between two price series.

        Parameters:
        price1 (pd.Series): The first price series.
        price2 (pd.Series): The second price series.

        Returns:
        str: A message indicating whether the series are cointegrated or not.
        """
        # Combine the two price series into a DataFrame
        df = pd.concat([price1, price2], axis=1)
        df.columns = ['Price1', 'Price2']

        # Perform Engle-Granger cointegration test
        eg_test_result = sm.tsa.coint(df['Price1'], df['Price2'])
        result_message = 'Engle-Granger Cointegration Test:\n'
        result_message += 'Cointegration t-statistic: {}\n'.format(eg_test_result[0])
        result_message += 'Cointegration p-value: {}\n'.format(eg_test_result[1])
        result_message += 'Critical Values:\n'
        for i, crit_val in enumerate(eg_test_result[2]):
            result_message += f'\tCritical Value {i}: {crit_val}\n'

        # Conclusion
        if eg_test_result[1] <= 0.05:
            result_message += 'The series are cointegrated.'
        else:
            result_message += 'The series are not cointegrated.'

        print(result_message)