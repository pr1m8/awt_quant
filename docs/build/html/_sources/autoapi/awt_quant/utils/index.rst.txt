awt_quant.utils
===============

.. py:module:: awt_quant.utils




Module Contents
---------------

.. py:function:: hurst(ts, lag)

   Calculates the Hurst Exponent for a given time series.

   The Hurst Exponent is a measure of long-term memory in a time series:
   - Near 0.5: Random series.
   - Near 0: Mean reverting.
   - Near 1: Trending.

   :param ts: Time series data.
   :type ts: array-like
   :param lag: Maximum lag to compute.
   :type lag: int

   :returns: Hurst exponent value.
   :rtype: float


.. py:function:: hurst(ts, lag)

   Returns the Hurst Exponent of the time series vector ts
   The Hurst Exponent is a statistical measure used to classify time series and infer the level of difficulty in predicting and
   choosing an appropriate model for the series at hand. The Hurst exponent is used as a measure of long-term memory of time series.
   It relates to the autocorrelations of the time series, and the rate at which these decrease as the lag between pairs of
   values increases.

   Value near 0.5 indicates a random series.
   Value near 0 indicates a mean reverting series.
   Value near 1 indicates a trending series.


.. py:function:: financial_calendar_days_before(date_str, T, calendar_name='NYSE')

   Gets the T-th market day occurring before a given date.

   :param date_str: End date in 'YYYY-MM-DD' format.
   :type date_str: str
   :param T: Number of market days to go back.
   :type T: int
   :param calendar_name: Market calendar name (default: 'NYSE').
   :type calendar_name: str

   :returns: Computed start date in 'YYYY-MM-DD' format.
   :rtype: str


.. py:function:: plot_correlogram(x, lags=None, title=None)

   Plots the correlogram for a given time series.

   The output consists of:
   - Time series plot.
   - Q-Q plot.
   - Autocorrelation Function (ACF).
   - Partial Autocorrelation Function (PACF).

   :param x: Time series data.
   :type x: pd.Series
   :param lags: Number of lags in ACF/PACF.
   :type lags: int, optional
   :param title: Plot title.
   :type title: str, optional


