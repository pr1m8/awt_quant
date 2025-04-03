awt_quant.forecast.stochastic.stochastic_models
===============================================

.. py:module:: awt_quant.forecast.stochastic.stochastic_models




Module Contents
---------------

.. py:class:: StochasticSimulator(num_paths, N, dt, device)

   .. py:attribute:: num_paths


   .. py:attribute:: N


   .. py:attribute:: dt


   .. py:attribute:: device


   .. py:method:: simulate_gbm(mu, sigma, S0)

      Simulates Geometric Brownian Motion (GBM).



   .. py:method:: estimate_ou_parameters(data)

      Estimates Ornstein-Uhlenbeck (OU) process parameters via MLE.



   .. py:method:: simulate_ou(S0, data)

      Simulates Ornstein-Uhlenbeck (OU) process with estimated parameters.



   .. py:method:: estimate_cir_parameters(data)

      Estimates Cox-Ingersoll-Ross (CIR) process parameters via MLE.



   .. py:method:: simulate_cir(S0, data)

      Simulates Cox-Ingersoll-Ross (CIR) process with estimated parameters.



   .. py:method:: estimate_heston_parameters(cond_vol, returns)

      Estimates parameters for Heston model.



   .. py:method:: simulate_heston(S0, cond_vol, returns)

      Simulates the Heston model.



