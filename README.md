# AWT Quant

**AWT Quant** is a next-generation quantitative research and financial modeling platform built for portfolio optimization, forecasting, risk analysis, and macroeconomic insight. It blends traditional stochastic modeling with state-of-the-art large language models (LLMs) and automated agent tooling.

> 💹 From SPDE Monte Carlo simulations to AutoGPT-powered research assistants and macroeconomic database access — **AWT Quant** is your full-stack quant toolkit.

---

## 🌟 Summary

AWT Quant supports:

- **Stochastic PDE Simulations** (e.g., GBM, Heston, CIR, OU, MJD)
- **Scenario and Stress Testing**
- **Portfolio Optimization** with forecastable constraints
- **Portfolio and Macro Forecasting** (LLM + SPDE-powered)
- **Lag-Llama-based LLM Forecasting** ([Hugging Face Model](https://huggingface.co/time-series-foundation-models/Lag-Llama))
- **Access to 800k+ macroeconomic time series**
- **Integration with AutoGPT-style agents** for research and evaluation
- **Realistic asset simulation with jump diffusion, volatility clustering, and stochastic drift**
- **Backtestable pipelines & dynamic strategy analysis**

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
from awt_quant.simulators import SPDEMCSimulator

# Initialize simulator
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

## 🧠 Forecasting & Macro Intelligence

AWT Quant features:

- 📊 LLM + Time Series forecasting (Lag-Llama, Prophet, and custom)
- 🌍 Access 800k+ global macroeconomic series
- 🔗 Integrate with [FRED](https://fred.stlouisfed.org/), OECD, IMF, World Bank
- 🧮 Combine scenario assumptions with forward simulations

```python
from awt_quant.llm.forecasting import LagLlamaForecaster

model = LagLlamaForecaster("AAPL")
forecast_df = model.forecast(horizon=60)
model.plot()
```

---

## 🧬 Supported Stochastic Models

| Model | Description |
|-------|-------------|
| **GBM** | Standard model for stock price movements, assumes log-normal returns |
| **Heston** | Captures stochastic volatility using a two-factor model |
| **CIR** | Square-root mean-reverting process (popular for interest rates) |
| **Ornstein-Uhlenbeck** | Classic mean-reverting process for signals and spreads |
| **MJD** | Merton Jump Diffusion – introduces jumps in asset price dynamics |

---

## 📚 Use Cases

### 🧠 Forecasting
- Asset prices, volatility, macro indicators, yield curves
- LLMs + SPDE ensembles for predictive modeling

### 💼 Portfolio Optimization
- Constrained optimization with forecasts
- CVaR, Black-Litterman, or expected return-based frameworks

### 🧪 Stress Testing
- Hypothetical market conditions
- Recession / Inflation / Policy Shocks

### 🧠 AutoGPT Integration
- Plan and run financial research autonomously
- Agents can:
  - Query macro series
  - Simulate portfolios
  - Generate reports or insights

---

## 🧠 Example: Portfolio Forecasting + Optimization

```python
from awt_quant.optimization import PortfolioForecaster

forecaster = PortfolioForecaster(tickers=["AAPL", "MSFT", "TSLA"])
forecaster.forecast(horizon=30)
opt_result = forecaster.optimize(max_volatility=0.1)
forecaster.plot()
```

---

## 📁 Directory Structure

```bash
awt_quant/
├── simulators/               # SPDE models (gbm.py, heston.py, etc.)
├── llm/                      # LLM-powered forecasting tools
├── optimization/            # Portfolio optimizers
├── macro/                   # Macro data connectors
├── agents/                  # AutoGPT-style financial agents
├── utils/                   # Helper utilities
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

Full documentation available at [awt-quant.readthedocs.io](https://awt-quant.readthedocs.io/)

- API reference
- Model tutorials
- Portfolio tools
- Agent automation
- Macro API integration

---

## 💡 Inspiration & Foundation

- [Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama)
- [QuantEcon](https://quantecon.org/)
- [AutoGPT](https://github.com/Torantulino/Auto-GPT)
- [Black-Litterman Model](https://www.investopedia.com/terms/b/blacklittermanmodel.asp)

---

## 📘 License

MIT License © 2025 William R. Astley / Pr1m8
