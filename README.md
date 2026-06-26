# Solvability of Differential Riccati Equations and Applications to Algorithmic Trading with Signals

This repository contains the Python/PyTorch implementation of the research paper titled **"Solvability of Differential Riccati Equations and Applications to Algorithmic Trading with Signals"** by Fayçal Drissi. The paper introduces and studies a differential Riccati equation (DRE) with indefinite matrix coefficients and demonstrates its application in algorithmic trading problems. 

The implementation provided here focuses on solving the DRE and applying it to two key algorithmic trading strategies described in the paper:
1. **Multi-Asset Market Making Strategy**: A strategy for over-the-counter markets where a market maker uses an external trading venue to hedge risk.
2. **Optimal Trading Strategy with Signals**: A strategy that utilizes asset prices and external signals to learn the drift in asset prices and make informed trading decisions.

## Core Concept

The **Differential Riccati Equation (DRE)** is a type of matrix differential equation that arises in control theory and optimization problems. In this work, the author explores its solvability under conditions of indefinite matrix coefficients, which are critical in practical scenarios such as financial modeling. The paper establishes the connection between solving the DRE and optimal control problems, providing key insights into existence and uniqueness of solutions.

### Applications in Algorithmic Trading

The paper leverages the DRE to formulate and solve two algorithmic trading problems:
1. **Multi-Asset Market Making**: The market maker optimizes their position by balancing inventory and hedging risks using an external trading venue.
2. **Drift Learning for Optimal Trading**: Traders use past price observations and external signals to learn the drift in asset prices, enabling them to execute optimal trades.

Both problems are approached using the framework of constant absolute risk-aversion (CARA) utility functions.

## Repository Structure

The repository includes Python and PyTorch code to implement the concepts and algorithms described in the paper. Below is a breakdown of the key components:

### 1. `differential_riccati_solver.py`
This module contains the implementation of the solver for the differential Riccati equation (DRE). It uses numerical techniques to compute the solution given matrix coefficients and initial conditions.

### 2. `market_making_strategy.py`
This script implements the multi-asset market making strategy described in the paper. It models the market maker's inventory management and risk mitigation via an external hedging venue.

### 3. `drift_learning_strategy.py`
This module implements the optimal trading strategy based on drift learning. It uses historical price data and external signals to estimate the drift in asset prices and derive optimal trading actions.

### 4. `utils.py`
Contains utility functions for matrix operations, numerical integration, and other helper methods used across the implementation.

### 5. `examples/`
This folder includes example scripts that demonstrate how to use the provided modules to simulate trading strategies and solve sample DREs.

### 6. `README.md`
The current file, which provides an overview of the repository and its contents.

## Requirements

To run the code, the following dependencies are required:
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib (for visualization)

You can install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Usage

### Solving a Differential Riccati Equation
To solve a Differential Riccati Equation using the provided solver:
```python
from differential_riccati_solver import solve_dre

# Example: Define matrix coefficients and initial conditions
A = ...
B = ...
C = ...
X0 = ...

# Solve the DRE
solution = solve_dre(A, B, C, X0)
```

### Running the Market Making Strategy
To simulate the multi-asset market making strategy:
```python
from market_making_strategy import simulate_market_making

# Example: Define market parameters and initial conditions
params = {...}
initial_inventory = ...

# Run the simulation
results = simulate_market_making(params, initial_inventory)
```

### Drift Learning for Optimal Trading
To execute the optimal trading strategy using drift learning:
```python
from drift_learning_strategy import optimal_trading

# Example: Define price history and signal data
price_data = ...
signal_data = ...

# Run the trading strategy
trades = optimal_trading(price_data, signal_data)
```

## Directory Structure

```
├── differential_riccati_solver.py
├── market_making_strategy.py
├── drift_learning_strategy.py
├── utils.py
├── examples/
│   ├── example_market_making.py
│   ├── example_drift_learning.py
├── requirements.txt
├── README.md
```

## References

- Fayçal Drissi, *Solvability of Differential Riccati Equations and Applications to Algorithmic Trading with Signals*. [arXiv:2202.07478](https://arxiv.org/pdf/2202.07478v3)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or suggestions, please contact the author of the paper or open an issue in this repository.