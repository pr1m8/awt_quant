# AWT Quant

**AWT Quant** is a next-generation quantitative research and financial modeling platform built for portfolio optimization, forecasting, risk analysis, and macroeconomic insight. It blends traditional stochastic modeling with state-of-the-art large language models (LLMs), stress testing frameworks, and automated agent tooling.

> 💹 From SPDE Monte Carlo simulations to TimeGPT-powered research assistants, stress testing, portfolio optimization, and macroeconomic database access — **AWT Quant** is your full-stack quant toolkit.

---

## 🌟 Summary

AWT Quant supports:

- **Stochastic PDE Simulations** (e.g., GBM, Heston, CIR, OU, MJD)
- **Stress Testing & Scenario Forecasting** (macro + portfolio)
- **Portfolio Optimization** with forecastable constraints and LLM integration
- **Portfolio & Macro Forecasting** (SPDE + Lag-Llama)
- **Lag-Llama-based LLM Forecasting** ([Hugging Face Model](https://huggingface.co/time-series-foundation-models/Lag-Llama))
- **Access to 800k+ macroeconomic time series**
- **Risk & Performance Reporting** with Tearsheet generation
- **TimeGPT-style agent pipelines** for simulation, evaluation, and reporting
- **Realistic asset simulation with jump diffusion, volatility clustering, and stochastic drift**
- **Backtestable workflows, modular pipelines, and multi-layered inference**

---

## 📦 Installation

```bash
pip install awt-quant
# or using poetry
poetry add awt-quant
```

---

## 📈 Getting Started

```python
from awt_quant.forecast.stochastic.run_simulations import SPDEMCSimulator

sim = SPDEMCSimulator(
    symbol='AAPL',
    start_date='2022-01-01',
    end_date='2022-03-01',
    dt=1,
    num_paths=1000,
    eq='heston'  # Supports 'gbm', 'heston', 'cir', 'ou', 'mjd'
)

sim.download_data()
sim.set_parameters()
sim.simulate()
sim.plot_simulation()
```

---

## 🧐 Forecasting & Macro Intelligence

AWT Quant features:

- 📊 LLM + Time Series forecasting (Lag-Llama, GARCH, Macro)
- 🌍 Access 800k+ global macroeconomic series
- 🔗 Integrate with FRED, IMF, World Bank, OECD
- 🧼 Combine scenario assumptions with forward simulations

```python
from awt_quant.forecast.lag_llama_forecast import LagLlamaForecaster

model = LagLlamaForecaster("AAPL")
forecast_df = model.forecast(horizon=60)
model.plot()
```

---

## 🧬 Supported Stochastic Models

| Model | Description |
|-------|-------------|
| **GBM** | Log-normal price movement (Black-Scholes) |
| **Heston** | Stochastic volatility with mean reversion |
| **CIR** | Interest rate modeling with mean-reverting variance |
| **OU** | Signal modeling with noise around a mean |
| **MJD** | GBM + discrete jumps (shock scenarios) |

---

## 📚 Use Cases

### 🧐 Forecasting
- Assets, volatility, macro, yields
- Ensemble LLMs + SPDE pipelines

### 💼 Portfolio Optimization
- Constraints: Volatility, CVaR, allocation
- Black-Litterman & forecast-driven engines

### 🧪 Stress Testing
- Regime shifts, interest rate shocks
- Economic downturns and inflation shocks

### 📊 Risk Analytics
- Generate custom tearsheets
- Performance attribution and statistical risk factors

### 🤖 Autonomous Research Agents
- Powered by TimeGPT-style pipelines
- Simulate, analyze, report — automatically

---

## 🧐 Example: Portfolio Forecasting + Optimization

```python
from awt_quant.forecast.portfolio.portfolio_forecast import PortfolioForecaster

forecaster = PortfolioForecaster(["AAPL", "MSFT", "TSLA"])
forecaster.forecast(horizon=30)
opt_result = forecaster.optimize(max_volatility=0.1)
forecaster.plot()
```

---

## 📁 Directory Structure

```text
awt_quant/
├── data_fetch/
│   ├── macro.py
│   └── yf_fetch.py
├── forecast/
│   ├── garch_forecast.py
│   ├── lag_llama_forecast.py
│   ├── macro_forecast.py
│   ├── portfolio/
│   │   ├── portfolio_forecast.py
│   │   └── portfolio_simulations.py
│   └── stochastic/
│       ├── pde_forecast.py
│       ├── portfolio/
│       │   ├── portfolio_forecast.py
│       │   └── portfolio_simulations.py
│       ├── run_simulations.py
│       ├── stochastic_likelihoods.py
│       └── stochastic_models.py
├── portfolio/
│   ├── multi_factor_analysis/
│   │   ├── DataCollector.py
│   │   ├── FactorConstructor.py
│   │   ├── KMeansClusterer.py
│   │   ├── LocalizedModel.py
│   │   ├── RandomForestFeatureSelector.py
│   │   ├── StressSensitivityAnalysis.py
│   │   └── main.py
│   └── optimization/
│       └── optimize.py
├── risk/
│   └── tearsheet.py
├── utils.py
└── __init__.py
```

---

## 🛠 Development

```bash
git clone https://github.com/pr1m8/awt-quant.git
cd awt-quant
poetry install
```

---

## 📖 Documentation

[https://awt-quant.readthedocs.io](https://awt-quant.readthedocs.io)

Includes:
- API Reference
- Simulation notebooks
- Portfolio modeling guides
- Macro pipelines
- LLM integrations
- Risk analytics & tearsheet customization

---

## 📘 License

MIT License © 2025 William R. Astley / Pr1m8
