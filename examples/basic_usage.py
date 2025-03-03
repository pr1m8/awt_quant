"""
Basic usage example of the SPDE Monte Carlo Simulator.

This example demonstrates how to:
1. Initialize the simulator
2. Download historical data
3. Run simulations for different models
4. Plot the results
"""

from awt_quant.spdemc import SPDEMCSimulator
import matplotlib.pyplot as plt

def main():
    # Parameters
    symbol = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    dt = 1
    num_paths = 1000

    # Available models
    models = ['gbm', 'heston', 'cir', 'ou', 'mjd']

    # Create subplots for each model
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 15))
    fig.suptitle('SPDE Monte Carlo Simulations')

    # Run simulations for each model
    for i, model in enumerate(models):
        # Initialize simulator
        sim = SPDEMCSimulator(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            dt=dt,
            num_paths=num_paths,
            eq=model
        )

        # Run simulation
        sim.download_data()
        sim.set_parameters()
        sim.simulate()

        # Plot results
        axes[i].plot(sim.S[0, :])
        axes[i].set_title(f'{model.upper()} Model')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Price')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 