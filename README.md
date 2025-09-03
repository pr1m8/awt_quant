<!-- PROJECT SHIELDS -->
<p align="center">
  <a href="https://pypi.org/project/awt-quant/">
    <img src="https://img.shields.io/pypi/v/awt-quant?color=blue&label=PyPI" alt="PyPI"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/>
  </a>
  <a href="https://awt-quant.readthedocs.io">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Docs"/>
  </a>
  <a href="https://github.com/pr1m8/awt-quant/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/pr1m8/awt-quant/tests.yml?label=tests" alt="Tests"/>
  </a>
</p>

<br/>

# 🌐 AWT Quant  

**AWT Quant** is a **next-generation quantitative research & financial modeling platform**.  
It combines the best of **stochastic modeling**, **LLMs**, **portfolio analysis**, and **autonomous agents**  
to deliver a **comprehensive quant toolkit** for researchers, analysts, and traders.  

> 💹 From **SPDE Monte Carlo** simulations to **TimeGPT-style assistants**, portfolio optimization, stress testing, and macroeconomic integration — AWT Quant covers it all.  

---

## 📖 Table of Contents
- [✨ Features](#-features)
- [⚡️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📊 Modules](#-modules)
- [🧠 LLM Forecasting](#-llm-forecasting)
- [🧪 Use Cases](#-use-cases)
- [🛠 Development](#-development)
- [📖 Documentation](#-documentation)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

✔️ **Stochastic Models**: GBM, Heston, CIR, OU, MJD  
✔️ **Forecasting Engines**: GARCH, SPDE, LLM-based, macro-informed  
✔️ **Portfolio Optimization**: Constraints, objectives, efficient frontier visualization  
✔️ **Risk Analytics**: Stress testing, VaR, CVaR, tearsheets  
✔️ **Multi-Factor Analysis**: PCA, clustering, attribution, localized models  
✔️ **Macroeconomic Data**: 800k+ time series from FRED, IMF, WorldBank, OECD  
✔️ **Agent Workflows**: Autonomous research, trading, portfolio management  

---

## ⚡️ Installation  

```bash
pip install awt-quant
# or with poetry
poetry add awt-quant
```

---

## 🚀 Quick Start  

### 🎲 Stochastic Simulations  

```python
from awt_quant.forecast.stochastic.run_simulations import SPDEMCSimulator

sim = SPDEMCSimulator(
    symbol="AAPL",
    start_date="2022-01-01",
    end_date="2022-03-01",
    dt=1,
    num_paths=1000,
    eq="heston"
)

sim.download_data()
sim.set_parameters()
sim.simulate()
sim.plot_simulation()
```

### 📈 Portfolio Forecasting & Optimization  

```python
from awt_quant.forecast.portfolio.portfolio_forecast import PortfolioForecaster
from awt_quant.portfolio.optimization.optimize import PortfolioOptimizer

forecaster = PortfolioForecaster(["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"])
forecast_results = forecaster.forecast(horizon=30)

optimizer = PortfolioOptimizer(
    assets=["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"],
    objective="sharpe",
    constraints={"max_volatility": 0.15, "max_per_asset": 0.25},
    forecast_data=forecast_results
)

weights = optimizer.optimize()
optimizer.plot_efficient_frontier()
optimizer.plot_allocation()
```

---

## 📊 Modules  

| Module      | Description |
|-------------|-------------|
| **Stochastic Models** | GBM, Heston, CIR, OU, MJD |
| **Forecasting** | LLMs, GARCH, SPDE, macro forecasting |
| **Portfolio** | Optimization & multi-factor analysis |
| **Risk** | Tearsheets, stress tests, attribution |
| **Agents** | Research, forecasting, portfolio workflows |
| **Data** | Market & macro data acquisition |

---

## 🧠 LLM Forecasting  

- **Lag-Llama** → Time series foundation model with macro context  
- **TimeGPT Assistant** → Forecasts with narratives & natural language queries  
- **Macro Models** → Granger causality, transfer entropy, PCMCI  

---

## 🧪 Use Cases  

- 🔮 **Forecasting**: Ensemble statistical + stochastic + LLM methods  
- 💼 **Portfolio Optimization**: Forecast-informed allocation, constraints, Black-Litterman  
- 🔍 **Multi-Factor Analysis**: Clustering, attribution, sensitivity analysis  
- 📊 **Risk Analytics**: VaR, CVaR, drawdowns, stress tests  
- 🤖 **Autonomous Agents**: Research reports, trading strategies, monitoring  

---

## 🛠 Development  

```bash
git clone https://github.com/pr1m8/awt-quant.git
cd awt-quant
poetry install
pytest tests/
```

---

## 📖 Documentation  

👉 [**awt-quant.readthedocs.io**](https://awt-quant.readthedocs.io)  

Includes:  
- API reference  
- Interactive notebooks  
- Tutorials & guides  
- Case studies  
- Benchmarks  

---

## 📄 License  

MIT © 2025 [William R. Astley / Pr1m8](https://github.com/pr1m8)  

---

## 🙏 Acknowledgements  

- [Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama)  
- [AutoTS](https://github.com/winedarksea/AutoTS)  
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)  
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/)  
