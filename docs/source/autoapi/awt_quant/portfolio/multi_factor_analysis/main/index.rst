awt_quant.portfolio.multi_factor_analysis.main
==============================================

.. py:module:: awt_quant.portfolio.multi_factor_analysis.main






Module Contents
---------------

.. py:function:: run_multi_factor_analysis(symbols, start_date='2020-01-01', end_date='2022-01-01')

   Runs the Multi-Factor Analysis (MFA) pipeline.

   :param symbols: List of stock tickers to analyze.
   :type symbols: list
   :param start_date: Start date for collecting financial data.
   :type start_date: str
   :param end_date: End date for collecting financial data.
   :type end_date: str

   :returns:

             Processed results including clustering, feature importance,
                   stress tests, and sensitivity analysis.
   :rtype: dict


.. py:data:: symbols
   :value: ['AAPL', 'MSFT', 'GOOGL', 'RTX', 'LMT', 'BA', 'FANG', 'AMZN', 'TSLA', 'JPM', 'GS', 'JNJ', 'PFE',...


