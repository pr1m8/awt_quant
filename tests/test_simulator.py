"""
Tests for the SPDE Monte Carlo Simulator.
"""

import pytest
import numpy as np
from spde_mc_simulator import SPDEMCSimulator

@pytest.fixture
def simulator():
    """Create a basic simulator instance for testing."""
    return SPDEMCSimulator(
        symbol='AAPL',
        start_date='2022-01-01',
        end_date='2022-03-01',
        dt=1,
        num_paths=100,
        eq='gbm'
    )

def test_initialization(simulator):
    """Test simulator initialization."""
    assert simulator.symbol == 'AAPL'
    assert simulator.dt == 1
    assert simulator.num_paths == 100
    assert simulator.eq == 'gbm'

def test_data_download(simulator):
    """Test data downloading functionality."""
    simulator.download_data()
    assert simulator.training_data is not None
    assert simulator.act_data is not None
    assert simulator.S0 is not None
    assert 'Close' in simulator.training_data.columns

def test_parameter_setting(simulator):
    """Test parameter setting functionality."""
    simulator.download_data()
    simulator.set_parameters()
    assert simulator.mu is not None
    assert simulator.sigma is not None
    assert simulator.T is not None
    assert simulator.N is not None

def test_simulation_shape(simulator):
    """Test simulation output shape."""
    simulator.download_data()
    simulator.set_parameters()
    simulator.simulate()
    assert simulator.S is not None
    assert simulator.S.shape[0] == simulator.num_paths
    assert simulator.S.shape[1] >= simulator.N

def test_different_models():
    """Test all available models."""
    models = ['gbm', 'heston', 'cir', 'ou', 'mjd']
    for model in models:
        sim = SPDEMCSimulator(
            symbol='AAPL',
            start_date='2022-01-01',
            end_date='2022-03-01',
            dt=1,
            num_paths=100,
            eq=model
        )
        sim.download_data()
        sim.set_parameters()
        sim.simulate()
        assert sim.S is not None
        assert not np.isnan(sim.S).any() 