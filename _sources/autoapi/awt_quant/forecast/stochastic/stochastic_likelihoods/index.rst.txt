awt_quant.forecast.stochastic.stochastic_likelihoods
====================================================

.. py:module:: awt_quant.forecast.stochastic.stochastic_likelihoods

.. autoapi-nested-parse::

   Negative Log-Likelihood Functions for Stochastic Models

   This module provides:
       - neg_log_likelihood_ou: Log-likelihood for Ornstein-Uhlenbeck (OU) process.
       - neg_log_likelihood_cir: Log-likelihood for Cox-Ingersoll-Ross (CIR) process.





Module Contents
---------------

.. py:function:: neg_log_likelihood_ou(params, data, dt)

   Computes the negative log-likelihood for the Ornstein-Uhlenbeck (OU) process.

   :param params: (mu, kappa, sigma) parameters.
   :type params: tuple
   :param data: Log-returns data.
   :type data: np.ndarray
   :param dt: Time step.
   :type dt: float

   :returns: Negative log-likelihood value.
   :rtype: float


.. py:function:: neg_log_likelihood_cir(params, data, dt)

   Computes the negative log-likelihood for the Cox-Ingersoll-Ross (CIR) process.

   :param params: (kappa, theta, sigma) parameters.
   :type params: tuple
   :param data: Volatility data.
   :type data: np.ndarray
   :param dt: Time step.
   :type dt: float

   :returns: Negative log-likelihood value.
   :rtype: float


