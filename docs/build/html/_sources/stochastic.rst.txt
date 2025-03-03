# **Stochastic Partial Differential Equation (PDE) Forecasting**

## **Overview**
This module implements **stochastic differential equations (SDEs) and PDE-based models** for forecasting **financial time series, volatility, and option pricing**. These models extend standard time-series forecasting by incorporating **stochastic volatility, mean reversion, and random noise components**.

### **Features**
- ✅ **GARCH-based volatility forecasting**  
- ✅ **Heston model for stochastic volatility**  
- ✅ **Cox-Ingersoll-Ross (CIR) mean-reverting volatility**  
- ✅ **Ornstein-Uhlenbeck (OU) process for interest rates & price modeling**  
- ✅ **Monte Carlo simulation for path forecasting**  
- ✅ **Conditional volatility forecasting using GARCH & MLE optimization**  
- ✅ **Support for asset price simulation, option pricing, and risk estimation**  

---

## **Module Structure**
```
stochastic_forecast/
│── __init__.py
│── pde_forecast.py            # SPDEMCSimulator: PDE-based forecasting model
│── portfolio_forecast.py       # Monte Carlo simulation for portfolio forecasting
│── stochastic_models.py        # Core SDE-based models (Heston, CIR, GBM, OU)
│── stochastic_likelihoods.py   # Likelihood estimation for model fitting
│── run_simulations.py          # Runs multiple simulations for comparison
```

---

## **Key Components**
### **1. `pde_forecast.py` (SPDEMCSimulator)**
This file contains the `SPDEMCSimulator` class, which models stock prices using **stochastic PDEs** such as the **Heston, Geometric Brownian Motion (GBM), CIR, and Ornstein-Uhlenbeck (OU) processes**.

#### **How It Works**
1. **Download stock price data** using Yahoo Finance (`yf_fetch.py`).
2. **Estimate volatility dynamics** using:
   - **GARCH model** for conditional volatility estimation.
   - **MLE optimization** for CIR/OU parameters.
3. **Simulate stochastic processes** for stock prices.
4. **Monte Carlo simulation** to generate future price paths.
5. **Plot forecasts** using quantiles and confidence intervals.

#### **Available Models**
| Model | Equation | Use Case |
|--------|----------|----------|
| **GBM (Geometric Brownian Motion)** | \( dS_t = \mu S_t dt + \sigma S_t dW_t \) | Basic stochastic price movement |
| **Heston Model** | \( dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^1 \), \( dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^2 \) | Stochastic volatility model |
| **CIR (Cox-Ingersoll-Ross)** | \( dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t \) | Mean-reverting volatility model |
| **OU (Ornstein-Uhlenbeck)** | \( dX_t = \kappa (\theta - X_t) dt + \sigma dW_t \) | Interest rate modeling |

#### **Example Usage**
```python
from awt_quant.forecast.stochastic_forecast import SPDEMCSimulator

# Initialize Simulator
sim = SPDEMCSimulator("AAPL", "Heston", start_date="2023-01-01", end_date="2023-10-01", num_paths=1000)

# Download data and estimate parameters
sim.download_data(train_test_split=0.75)
sim.simulate("Heston")

# Plot simulation results
sim.plot_simulation("Heston")
```

---

### **2. `stochastic_models.py`**
This module contains **core SDE implementations** used in `SPDEMCSimulator`. It includes functions for **simulating asset paths**, **estimating parameters**, and **generating sample paths**.

#### **Functions**
- `simulate_gbm(S0, mu, sigma, dt, N, num_paths)`: Simulates **Geometric Brownian Motion (GBM)**.
- `simulate_heston(S0, v0, mu, kappa, theta, sigma, rho, dt, N, num_paths)`: Simulates **Heston Model**.
- `simulate_cir(v0, kappa, theta, sigma, dt, N, num_paths)`: Simulates **Cox-Ingersoll-Ross Process**.
- `simulate_ou(X0, kappa, theta, sigma, dt, N, num_paths)`: Simulates **Ornstein-Uhlenbeck Process**.

#### **Example Usage**
```python
from awt_quant.forecast.stochastic_forecast.stochastic_models import simulate_gbm

# Simulate GBM paths
gbm_paths = simulate_gbm(S0=100, mu=0.05, sigma=0.2, dt=1/252, N=252, num_paths=1000)
```

---

### **3. `stochastic_likelihoods.py`**
This module provides **likelihood estimation functions** for stochastic models, allowing parameter estimation using **Maximum Likelihood Estimation (MLE)**.

#### **Functions**
- `neg_log_likelihood_ou(params, data)`: Computes **negative log-likelihood** for **Ornstein-Uhlenbeck** process.
- `neg_log_likelihood_cir(params, data)`: Computes **negative log-likelihood** for **Cox-Ingersoll-Ross** process.
- `optimize_parameters(series, model)`: **Optimizes parameters** for stochastic volatility models.

#### **Example Usage**
```python
from awt_quant.forecast.stochastic_forecast.stochastic_likelihoods import optimize_parameters

# Optimize CIR parameters for given volatility series
optimal_params = optimize_parameters(volatility_series, model="CIR")
```

---

### **4. `run_simulations.py`**
Provides a **wrapper function** to run multiple stochastic model simulations and compare results.

#### **How It Works**
- Runs **Monte Carlo simulations** for different stochastic models.
- **Compares** forecast errors for each model.
- **Plots** forecast results.

#### **Example Usage**
```python
from awt_quant.forecast.stochastic_forecast import run_stochastic_simulations

run_stochastic_simulations(ticker="AAPL", models=["GBM", "Heston", "CIR", "OU"], num_paths=1000)
```

---

## **Example: Comparing Different Stochastic Models**
```python
from awt_quant.forecast.stochastic_forecast import run_stochastic_simulations

run_stochastic_simulations(
    ticker="AAPL",
    models=["GBM", "Heston", "CIR", "OU"],
    num_paths=1000
)
```
✅ **Output:** A comparison of different stochastic models' predictions for **AAPL stock price**.

---

## **Key Takeaways**
- 🚀 **Stochastic PDE forecasting** extends traditional time-series forecasting with **stochastic volatility and mean-reversion**.
- ✅ **Supports multiple models:** GBM, Heston, CIR, and OU.
- ✅ **Parameter estimation** using MLE and GARCH.
- ✅ **Monte Carlo simulations** for uncertainty quantification.
- ✅ **Forecasting & risk modeling** for stocks and options.

---

## **Contributors & References**
- **Python Libraries:** `arch`, `scipy`, `numpy`, `pandas`, `torch`
- **Author:** [Your Name / Team Name]
- **License:** MIT License

