# AWT Quant

**AWT Quant** is a next-generation quantitative research and financial modeling platform that integrates traditional stochastic modeling with state-of-the-art large language models (LLMs), multi-factor analysis, and autonomous agent workflows. The platform provides comprehensive tools for portfolio optimization, forecasting, risk analysis, and macroeconomic insight.

> 💹 From SPDE Monte Carlo simulations to TimeGPT-style research assistants, stress testing, portfolio optimization, and macroeconomic database access — **AWT Quant** is your full-stack quantitative finance toolkit.

---

## 🌟 Core Capabilities

AWT Quant integrates multiple analytical paradigms:

- **Advanced Stochastic Modeling** (GBM, Heston, CIR, OU, MJD)
- **Multi-Modal Forecasting** combining statistical, LLM, and simulation approaches
- **Multi-Factor Portfolio Analysis** with clustering, PCA, and localized modeling
- **Portfolio Optimization** with customizable constraints and objectives
- **Comprehensive Risk Analytics** including stress testing and tearsheet generation
- **Macroeconomic Integration** with access to 800k+ time series datasets
- **Agent-Based Workflows** for autonomous research and strategy development

---

## 📦 Installation

```bash
pip install awt-quant
# or using poetry
poetry add awt-quant
```

---

## 📈 Getting Started

### Stochastic Simulations

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

### Portfolio Forecasting & Optimization

```python
from awt_quant.forecast.portfolio.portfolio_forecast import PortfolioForecaster
from awt_quant.portfolio.optimization.optimize import PortfolioOptimizer

# Forecast portfolio assets
forecaster = PortfolioForecaster(["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"])
forecast_results = forecaster.forecast(horizon=30)

# Optimize portfolio allocation with constraints
optimizer = PortfolioOptimizer(
    assets=["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"],
    objective="sharpe",  # Options: "sharpe", "min_volatility", "max_return", "min_cvar"
    constraints={
        "max_volatility": 0.15,
        "max_per_asset": 0.25,
        "min_per_asset": 0.05,
        "sectors": {
            "Technology": (0.3, 0.6),
            "Consumer Cyclical": (0.1, 0.3)
        }
    },
    forecast_data=forecast_results  # Optional: incorporate forecasts
)

weights = optimizer.optimize()
optimizer.plot_efficient_frontier()
optimizer.plot_allocation()
```

### Multi-Factor Analysis

```python
from awt_quant.portfolio.multi_factor_analysis.main import MultiFactorAnalysis

# Initialize analysis with portfolio assets and factors
mfa = MultiFactorAnalysis(
    assets=["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "META", "NFLX"],
    factors=["Market", "Size", "Value", "Momentum", "Quality"],
    start_date="2020-01-01"
)

# Run complete analysis pipeline
mfa.collect_data()
mfa.construct_factors()
mfa.run_clustering()
mfa.create_localized_models()
mfa.run_feature_selection()
mfa.run_stress_sensitivity()

# Generate insights and visualizations
factor_exposures = mfa.get_factor_exposures()
cluster_report = mfa.generate_cluster_report()
stress_scenarios = mfa.stress_test(scenarios=["inflation_shock", "rate_hike", "recession"])

mfa.plot_factor_attribution()
mfa.plot_clusters()
mfa.plot_stress_sensitivity()
```

### Advanced Forecasting With LLMs and Time Series Models

```python
from awt_quant.forecast.lag_llama_forecast import LagLlamaForecaster
from awt_quant.forecast.garch_forecast import GARCHForecaster
from awt_quant.forecast.macro_forecast import MacroForecaster
from awt_quant.forecast.stochastic.pde_forecast import SPDEForecaster

# Lag-Llama LLM-based forecasting with configurable parameters
llm_model = LagLlamaForecaster(
    symbol="AAPL",
    context_length=128,
    prediction_intervals=True,
    quantiles=[0.1, 0.5, 0.9],
    include_macro_context=True,
    macro_indicators=["UNRATE", "CPIAUCSL", "DFF"]
)
llm_forecast = llm_model.forecast(horizon=60)
llm_model.plot(include_history=True, plot_intervals=True)
llm_model.evaluate(metrics=["MSE", "MAE", "MAPE", "CRPS"])

# TimeGPT-style forecasting with narrative generation
from awt_quant.forecast.time_gpt_forecast import TimeGPTForecaster

time_gpt = TimeGPTForecaster(
    symbol="AAPL",
    frequency="daily",
    exogenous_variables=["VIX", "DXY", "US10Y"],
    generate_narrative=True
)
time_gpt_forecast = time_gpt.forecast(horizon=30)
time_gpt.plot()
time_gpt.generate_forecast_explanation()  # Natural language explanation of forecast

# GARCH volatility forecasting with custom specifications
garch = GARCHForecaster(
    symbol="AAPL",
    vol_model="EGARCH",  # Options: "GARCH", "EGARCH", "GJR-GARCH", "FIGARCH"
    mean_model="AR",     # Options: "Constant", "AR", "HAR", "ARMA"
    distribution="skewt"  # Options: "normal", "t", "skewt", "ged"
)
garch_result = garch.forecast(
    days=30,
    simulations=1000,
    return_volatility=True
)
garch.plot(include_forecast_intervals=True)

# Stochastic PDE forecasting with advanced configurations
spde = SPDEForecaster(
    symbol="AAPL",
    model="heston",  # Options: "gbm", "heston", "cir", "ou", "mjd"
    calibration_window="1Y",
    drift_estimation="kalman",  # Options: "mle", "kalman", "bayesian"
    jump_detection=True
)
spde_forecast = spde.forecast(
    horizon=60,
    paths=10000,
    quantiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    return_all_paths=True
)
spde.plot_forecast_fan()
spde.plot_paths(sample=100)
spde.analyze_tail_risk()

# Macro-based forecasting with causal inference
macro = MacroForecaster(
    target="AAPL",
    indicators=["GDP", "UNRATE", "CPIAUCSL", "DFF", "M2", "BAMLH0A0HYM2"],
    lag_window=12,
    causality_test="granger",  # Options: "granger", "transfer_entropy", "pcmci"
    feature_selection=True
)
macro_forecast = macro.forecast(
    horizon=30,
    method="elastic_net",  # Options: "elastic_net", "var", "prophet", "boosting"
    ensemble=True
)
macro.plot_relationships()
macro.plot_forecast(include_intervals=True)
macro.plot_impulse_response()
```

### Risk Analytics & Tearsheets

```python
from awt_quant.risk.tearsheet import RiskTearsheet

# Generate comprehensive risk report
tearsheet = RiskTearsheet(
    assets=["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"],
    weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    start_date="2020-01-01",
    benchmark="SPY"
)

# Basic metrics report
tearsheet.generate()

# Detailed analysis with custom metrics
tearsheet.generate(
    include_drawdowns=True,
    include_stress_tests=True,
    include_factor_attribution=True,
    custom_metrics=["omega_ratio", "sortino_ratio", "calmar_ratio"]
)

# Save report
tearsheet.save_pdf("portfolio_report.pdf")
```

### Macroeconomic Data Integration

```python
from awt_quant.data_fetch.macro import MacroDataFetcher

# Initialize macro data fetcher
macro_data = MacroDataFetcher(
    sources=["FRED", "IMF", "WorldBank", "OECD"],
    start_date="2018-01-01"
)

# Fetch specific indicators
inflation = macro_data.fetch("CPIAUCSL", source="FRED")
unemployment = macro_data.fetch("UNRATE", source="FRED")
gdp_growth = macro_data.fetch("A191RL1Q225SBEA", source="FRED")

# Search for relevant indicators
oil_indicators = macro_data.search("oil price")
interest_rate_indicators = macro_data.search("interest rate", source="FRED")

# Plot relationships
macro_data.plot_correlation_matrix(["CPIAUCSL", "UNRATE", "DFF", "GDP"])
macro_data.plot_time_series(["CPIAUCSL", "PPIACO"], title="Inflation Measures")
```

### Agent-Based Workflows

```python
from awt_quant.agents.research_agent import QuantResearchAgent
from awt_quant.agents.forecasting_agent import ForecastingAgent
from awt_quant.agents.portfolio_agent import PortfolioAgent

# Research agent for autonomous analysis
research = QuantResearchAgent(
    task="Analyze tech sector performance under rising interest rates",
    data_sources=["yfinance", "FRED", "news_api"],
    output_format="report"
)
research_report = research.run()

# Forecasting agent for time series prediction
forecaster = ForecastingAgent(
    symbols=["AAPL", "MSFT", "AMZN"],
    horizon=30,
    methods=["lag_llama", "garch", "auto_ts", "stochastic"],
    ensemble=True
)
forecast_results = forecaster.run()

# Portfolio management agent
portfolio = PortfolioAgent(
    initial_allocation={"AAPL": 0.2, "MSFT": 0.2, "AMZN": 0.2, "TSLA": 0.2, "BND": 0.2},
    rebalance_frequency="weekly",
    risk_tolerance="moderate",
    optimization_objective="sharpe"
)

# Run backtest
backtest_results = portfolio.backtest(
    start_date="2020-01-01",
    end_date="2022-12-31"
)

# Deploy live trading strategy
portfolio.deploy(
    broker="alpaca",
    schedule="daily",
    monitoring=True
)
```

---

## 🧬 Supported Stochastic Models

| Model      | Description                                         | Best For                        |
| ---------- | --------------------------------------------------- | ------------------------------- |
| **GBM**    | Log-normal price movement (Black-Scholes)           | Standard equity price modeling  |
| **Heston** | Stochastic volatility with mean reversion           | Options pricing, vol clustering |
| **CIR**    | Interest rate modeling with mean-reverting variance | Fixed income, rates modeling    |
| **OU**     | Signal modeling with noise around a mean            | Mean-reverting series, spreads  |
| **MJD**    | GBM + discrete jumps (shock scenarios)              | Crash scenarios, tail risk      |

---

## 📊 Multi-Factor Analysis Components

The multi-factor analysis module provides a comprehensive toolkit for factor-based portfolio analysis:

### Data Collection & Factor Construction

- **DataCollector**: Acquires price data, financial fundamentals, and macroeconomic indicators
- **FactorConstructor**: Builds and validates custom factors (Value, Momentum, Quality, Low-Vol, etc.)

### Analytical Components

- **KMeansClusterer**: Groups assets by factor exposures and characteristics
- **LocalizedModel**: Creates specialized models for distinct asset clusters
- **RandomForestFeatureSelector**: Identifies most important factors for return prediction
- **StressSensitivityAnalysis**: Tests factor and portfolio sensitivity to market shocks

---

## 🧠 LLM Integration for Forecasting

AWT Quant leverages the latest advances in time series LLMs for financial forecasting:

### Lag-Llama Integration

- State-of-the-art time series foundation model
- Contextual forecasting with macro variable integration
- Uncertainty quantification with distributional forecasts
- Multi-horizon predictions with confidence intervals

### TimeGPT-style Agent System

- Research assistant capabilities for data exploration
- Autonomous report generation
- Natural language query interface for time series analysis
- Forecast explanation and narrative generation

---

## 📚 Use Cases

### 🔮 Advanced Forecasting

- Financial time series with multiple horizons and confidence intervals
- Volatility forecasting with GARCH, stochastic volatility, and LLM methods
- Integrated macro and market forecasting with causal inference
- Ensemble forecasting combining statistical, ML, and LLM approaches

### 💼 Sophisticated Portfolio Optimization

- Mean-variance, Black-Litterman, and factor-based optimization
- Customizable constraints: volatility, CVaR, allocation limits, sector exposure
- Forecast-informed optimization incorporating future scenarios
- Rebalancing strategies with transaction cost modeling

### 🔍 Multi-Factor Analysis

- Factor construction, validation, and attribution
- Asset clustering and localized modeling
- Factor sensitivity and exposure analysis
- Custom factor development and backtesting

### 🧪 Comprehensive Stress Testing

- Historical scenario replays (e.g., 2008 Crisis, 2020 COVID Crash)
- Custom macro shock scenarios (inflation, rates, growth shocks)
- Monte Carlo simulation with realistic market dynamics
- Sensitivity analysis across factors and asset classes

### 📊 Risk Analytics

- Customizable risk tearsheets and performance attribution
- Factor-based risk decomposition
- Comprehensive risk metrics (VaR, CVaR, drawdowns, etc.)
- Relative performance vs. benchmarks and risk-adjusted returns

### 🤖 Autonomous Research & Trading

- TimeGPT-style research assistants for quantitative analysis
- Backtestable trading strategies with agent-based decision making
- Scheduled reporting and monitoring systems
- Anomaly detection and alert generation

---

## 📁 Project Structure

```text
awt_quant/
├── data_fetch/             # Data acquisition modules
│   ├── macro.py            # Macroeconomic data from FRED, IMF, etc.
│   └── yf_fetch.py         # Market data from Yahoo Finance
├── forecast/               # Forecasting modules
│   ├── garch_forecast.py   # GARCH volatility forecasting
│   ├── lag_llama_forecast.py  # LLM-based time series forecasting
│   ├── macro_forecast.py   # Macroeconomic-based forecasting
│   ├── portfolio/          # Portfolio-level forecasting
│   └── stochastic/         # Stochastic process simulations
├── portfolio/              # Portfolio construction & analysis
│   ├── multi_factor_analysis/  # Factor-based analytics
│   └── optimization/       # Portfolio optimization engines
├── risk/                   # Risk analytics
│   └── tearsheet.py        # Performance & risk reporting
├── agents/                 # Agent-based workflows
└── utils.py                # Utility functions
```

---

## 🛠 Development

```bash
git clone https://github.com/pr1m8/awt-quant.git
cd awt-quant
poetry install
```

### Running Tests

```bash
pytest tests/
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📖 Documentation

[https://awt-quant.readthedocs.io](https://awt-quant.readthedocs.io)

Our documentation includes:

- Comprehensive API reference
- Interactive tutorials and notebooks
- Detailed guides for each module
- Case studies and example workflows
- Performance benchmarks
- Installation and configuration guides

---

## 📄 License

MIT License © 2025 William R. Astley / Pr1m8

---

## 🙏 Acknowledgements

- [Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama) time series foundation model
- [AutoTS](https://github.com/winedarksea/AutoTS) for time series forecasting
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) for portfolio optimization inspiration
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/) for macroeconomic data
