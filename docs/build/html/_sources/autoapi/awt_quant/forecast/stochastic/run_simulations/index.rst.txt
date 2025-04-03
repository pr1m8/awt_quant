awt_quant.forecast.stochastic.run_simulations
=============================================

.. py:module:: awt_quant.forecast.stochastic.run_simulations

.. autoapi-nested-parse::

   Run Stock Forecast Simulations using SPDEMCSimulator.

   This script allows running single and multiple stock simulations with different configurations.

   Usage:
       python run_simulations.py --symbol AAPL --mode single
       python run_simulations.py --mode multi







Module Contents
---------------

.. py:data:: calendar
   :value: 'NYSE'


.. py:data:: end_dates
   :value: ['2023-10-13', '2022-08-10', '2019-06-02', '2021-02-02']


.. py:data:: forecast_periods
   :value: [14, 30, 60, 90, 180, 252]


.. py:data:: train_test_splits
   :value: [0.75]


.. py:data:: dt
   :value: 1


.. py:data:: num_paths
   :value: 1000


.. py:data:: num_sim
   :value: 100


.. py:data:: plot_vol
   :value: True


.. py:data:: plot_sim
   :value: False


.. py:data:: eq_classes
   :value: ['Heston']


.. py:data:: eq_class
   :value: 'Heston'


.. py:function:: run_single_simulation(symbol)

   Runs a single simulation for a given stock symbol.


.. py:function:: run_multiple_simulations(symbols)

   Runs multiple simulations across different stock symbols.


.. py:data:: parser

