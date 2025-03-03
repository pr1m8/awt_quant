"""
Negative Log-Likelihood Functions for Stochastic Models

This module provides:
    - neg_log_likelihood_ou: Log-likelihood for Ornstein-Uhlenbeck (OU) process.
    - neg_log_likelihood_cir: Log-likelihood for Cox-Ingersoll-Ross (CIR) process.
"""

import numpy as np
from scipy.special import iv


def neg_log_likelihood_ou(params, data, dt):
    """
    Computes the negative log-likelihood for the Ornstein-Uhlenbeck (OU) process.

    Args:
        params (tuple): (mu, kappa, sigma) parameters.
        data (np.ndarray): Log-returns data.
        dt (float): Time step.

    Returns:
        float: Negative log-likelihood value.
    """
    mu, kappa, sigma = params
    psi = (2 * kappa * np.exp(kappa * dt)) / (2 * kappa - (1 + np.exp(-kappa * dt)) * (2 * kappa * np.exp(kappa * dt) - 2 * kappa))
    density = (1 / (sigma * np.sqrt(2 * np.pi * dt * psi * (1 - np.exp(-2 * kappa * dt)))))
    density *= np.exp(
        -((data[1:] - data[:-1] * np.exp(-kappa * dt) - mu * (1 - np.exp(-kappa * dt))) ** 2) /
        (2 * sigma ** 2 * dt * psi * (1 - np.exp(-2 * kappa * dt)))
    )
    return -np.log(density).sum()


def neg_log_likelihood_cir(params, data, dt):
    """
    Computes the negative log-likelihood for the Cox-Ingersoll-Ross (CIR) process.

    Args:
        params (tuple): (kappa, theta, sigma) parameters.
        data (np.ndarray): Volatility data.
        dt (float): Time step.

    Returns:
        float: Negative log-likelihood value.
    """
    kappa, theta, sigma = params
    c = 2 * kappa / ((1 - np.exp(-kappa * dt)) * sigma ** 2)
    q = 2 * kappa * theta / sigma ** 2 - 1
    u = c * data[:-1] * np.exp(-kappa * dt)
    v = c * data[1:]
    density = c * np.exp(-u - v) * (v / u) ** (q / 2) * iv(q, 2 * np.sqrt(u * v))
    return -np.log(density).sum()
