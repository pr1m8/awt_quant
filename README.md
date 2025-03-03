# SPDE Monte Carlo Simulator

A Python package for simulating various stochastic partial differential equations commonly used in financial modeling.
https://huggingface.co/time-series-foundation-models/Lag-Llama
## Features

- Geometric Brownian Motion (GBM)
- Heston Model
- Cox-Ingersoll-Ross (CIR) Model
- Ornstein-Uhlenbeck (OU) Process
- Merton Jump Diffusion (MJD) Model

## Installation

```bash
poetry install
```

## Usage

```python
from spde_mc_simulator import SPDEMCSimulator

# Initialize simulator
simulator = SPDEMCSimulator(
    symbol='AAPL',
    start_date='2022-01-01',
    end_date='2022-03-01',
    dt=1,
    num_paths=1000,
    eq='gbm'
)

# Run simulation
simulator.download_data()
simulator.set_parameters()
simulator.simulate()
simulator.plot_simulation()
```

## Models

### Geometric Brownian Motion (GBM)
Standard model for stock price movements assuming log-normal distribution.

### Heston Model
Stochastic volatility model that captures volatility clustering.

### Cox-Ingersoll-Ross (CIR)
Mean-reverting square-root process, commonly used for interest rates.

### Ornstein-Uhlenbeck (OU)
Mean-reverting process useful for modeling mean-reverting financial quantities.

### Merton Jump Diffusion (MJD)
Extends GBM with jump components to capture sudden price movements.

## License

MIT 