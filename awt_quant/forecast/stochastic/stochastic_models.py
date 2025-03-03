import numpy as np
import torch
from scipy.optimize import minimize
from .stochastic_likelihoods import neg_log_likelihood_ou, neg_log_likelihood_cir  # Modular likelihoods

class StochasticSimulator:
    def __init__(self, num_paths, N, dt, device):
        self.num_paths = num_paths
        self.N = N
        self.dt = dt
        self.device = device

    def simulate_gbm(self, mu, sigma, S0):
        """Simulates Geometric Brownian Motion (GBM)."""
        dW = torch.normal(mean=0.0, std=np.sqrt(self.dt), size=(self.num_paths, self.N)).to(self.device)
        dS = ((mu - 0.5 * sigma ** 2) * self.dt + sigma * dW).to(self.device)
        return torch.exp(torch.log(S0) + dS.cumsum(1))

    def estimate_ou_parameters(self, data):
        """Estimates Ornstein-Uhlenbeck (OU) process parameters via MLE."""
        init_params = np.array([data.mean().detach(), 0.2, data.std().detach()])
        result = minimize(neg_log_likelihood_ou, init_params, args=(data[1:], self.dt), method="Nelder-Mead")
        return torch.tensor(result.x)

    def simulate_ou(self, S0, data):
        """Simulates Ornstein-Uhlenbeck (OU) process with estimated parameters."""
        mu, kappa, sigma = self.estimate_ou_parameters(data)
        dW = torch.normal(mean=0.0, std=np.sqrt(self.dt), size=(self.num_paths, self.N)).to(self.device)
        dX = -kappa * (S0 - mu) * self.dt + sigma * dW
        return S0 + dX.cumsum(1)

    def estimate_cir_parameters(self, data):
        """Estimates Cox-Ingersoll-Ross (CIR) process parameters via MLE."""
        init_params = np.array([0.1, data.mean().detach(), data.std().detach()])
        result = minimize(neg_log_likelihood_cir, init_params, args=(data, self.dt), method="Nelder-Mead")
        return torch.tensor(result.x)

    def simulate_cir(self, S0, data):
        """Simulates Cox-Ingersoll-Ross (CIR) process with estimated parameters."""
        kappa, theta, sigma = self.estimate_cir_parameters(data)
        dW = torch.normal(mean=0.0, std=np.sqrt(self.dt), size=(self.num_paths, self.N)).to(self.device)
        S = torch.zeros((self.num_paths, self.N)).to(self.device)
        S[:, 0] = S0
        for t in range(1, self.N):
            dS = kappa * (theta - S[:, t - 1]) * self.dt + sigma * torch.sqrt(S[:, t - 1]) * dW[:, t - 1]
            S[:, t] = torch.abs(S[:, t - 1] + dS)  # Ensure positivity
        return S

    def estimate_heston_parameters(self, cond_vol, returns):
        """Estimates parameters for Heston model."""
        n = len(returns)
        cond_var = torch.tensor(cond_vol).to(self.device) ** 2

        num_kappa = n**2 - 2*n + 1 + cond_var[1:].sum() * (1 / cond_var)[:-1].sum() - cond_var[:-1].sum() * (1 / cond_var)[:-1].sum() - (n - 1) * (cond_var[1:] / cond_var[:-1]).sum()
        den_kappa = (n**2 - 2*n + 1 - (cond_var[:-1].sum() * (1 / cond_var[:-1]).sum())) * self.dt
        kappa = num_kappa / den_kappa

        num_theta = (n - 1) * cond_var[1:].sum() - (cond_var[1:] / cond_var[:-1]).sum() * cond_var[:-1].sum()
        den_theta = num_kappa
        theta = num_theta / den_theta

        sigma = (((cond_var[1:] - cond_var[:-1]) / np.sqrt(cond_var[:-1])) - ((kappa * (theta - cond_var[:-1]) * self.dt) / np.sqrt(cond_var[:-1]))).std()

        # Compute correlation
        dW1 = (returns[1:] - (returns.mean() - 0.5 * cond_var[:-1]) * self.dt) / np.sqrt(cond_var[:-1])
        dW2 = (cond_var[1:] - cond_var[:-1] - kappa * (theta - cond_var[:-1]) * self.dt) / (sigma * np.sqrt(cond_var[:-1]))
        rho = (dW1 * dW2).sum() / (n * self.dt)

        return torch.tensor([kappa, theta, sigma, rho])

    def simulate_heston(self, S0, cond_vol, returns):
        """Simulates the Heston model."""
        kappa, theta, sigma, rho = self.estimate_heston_parameters(cond_vol, returns)
        dW1 = torch.normal(mean=0.0, std=np.sqrt(self.dt), size=(self.num_paths, self.N)).to(self.device)
        dW2 = rho * dW1 + torch.sqrt(1 - rho**2) * torch.normal(mean=0.0, std=np.sqrt(self.dt), size=(self.num_paths, self.N)).to(self.device)

        v = torch.zeros((self.num_paths, self.N)).to(self.device)
        v[:, 0] = torch.tensor(cond_vol[-1])  # Initial volatility
        S = torch.zeros((self.num_paths, self.N)).to(self.device)
        S[:, 0] = S0

        for t in range(1, self.N):
            v[:, t] = torch.abs(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * self.dt + sigma * torch.sqrt(v[:, t - 1]) * dW2[:, t])
            dS = (returns.mean() - 0.5 * v[:, t]) * self.dt + torch.sqrt(v[:, t]) * dW1[:, t]
            S[:, t] = S[:, t - 1] * torch.exp(dS)

        return S
