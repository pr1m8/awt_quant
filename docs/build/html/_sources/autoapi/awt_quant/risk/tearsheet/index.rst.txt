awt_quant.risk.tearsheet
========================

.. py:module:: awt_quant.risk.tearsheet






Module Contents
---------------

.. py:function:: compute_beta(portfolio_value_series, ticker='^GSPC')

   Calculate the beta of the portfolio against a benchmark index.

   Args:
   portfolio_value_series (pandas.Series): Time series data of portfolio values.
   ticker (str): Ticker symbol of the benchmark index. Default is S&P 500 ('^GSPC').

   Returns:
   float: Beta value of the portfolio.


.. py:function:: common_index(series1, series2)

   Returns the common index values of two series.

   Args:
   series1 (pandas.Series): The first series.
   series2 (pandas.Series): The second series.

   Returns:
   pandas.Index: The common index values of the two series.


.. py:function:: risk_tearSheet(data, time_input='2y', risk_free_rate=0.02, confidence_level=0.95, benchmark_ticker='^GSPC')

.. py:data:: data

