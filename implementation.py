import numpy as np
import torch
from torch import nn
from torch.optim import Adam

class DifferentialRiccatiEquationSolver:
    def __init__(self, A, B, Q, R, T, dt):
        """
        Initialize the solver for the Differential Riccati Equation (DRE).
        Args:
            A (torch.Tensor): System dynamics matrix (n x n).
            B (torch.Tensor): Input matrix (n x m).
            Q (torch.Tensor): State cost matrix (n x n).
            R (torch.Tensor): Control cost matrix (m x m).
            T (float): Time horizon.
            dt (float): Time step size.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.T = T
        self.dt = dt
        self.n = A.shape[0]
        self.m = B.shape[1]

    def solve(self):
        """
        Solve the DRE backward in time using a finite difference method.
        Returns:
            P (torch.Tensor): Solution of the DRE at each time step (time_steps x n x n).
        """
        time_steps = int(self.T / self.dt)
        P = torch.zeros((time_steps + 1, self.n, self.n))
        P[-1] = self.Q  # Terminal condition

        for t in reversed(range(time_steps)):
            Pt = P[t + 1]
            dPdt = -(self.A.T @ Pt + Pt @ self.A - Pt @ self.B @ torch.linalg.inv(self.R) @ self.B.T @ Pt + self.Q)
            P[t] = Pt - self.dt * dPdt

        return P

class OptimalTradingStrategy:
    def __init__(self, A, B, Q, R, T, dt, initial_state, signals):
        """
        Initialize the optimal trading strategy solver.
        Args:
            A, B, Q, R, T, dt: Parameters for the DRE solver.
            initial_state (torch.Tensor): Initial state of the system (n x 1).
            signals (torch.Tensor): External signals (time_steps x n).
        """
        self.dre_solver = DifferentialRiccatiEquationSolver(A, B, Q, R, T, dt)
        self.initial_state = initial_state
        self.signals = signals
        self.dt = dt

    def compute_optimal_strategy(self):
        """
        Compute the optimal trading strategy using the solution of the DRE.
        Returns:
            states (torch.Tensor): Optimal states over time (time_steps x n).
            controls (torch.Tensor): Optimal controls over time (time_steps x m).
        """
        P = self.dre_solver.solve()
        time_steps = P.shape[0] - 1
        states = torch.zeros((time_steps + 1, self.initial_state.shape[0]))
        controls = torch.zeros((time_steps, self.dre_solver.m))

        states[0] = self.initial_state.squeeze()

        for t in range(time_steps):
            Kt = torch.linalg.inv(self.dre_solver.R) @ self.dre_solver.B.T @ P[t]
            ut = -Kt @ states[t].unsqueeze(-1) + self.signals[t].unsqueeze(-1)
            controls[t] = ut.squeeze()
            states[t + 1] = states[t] + self.dt * (self.dre_solver.A @ states[t].unsqueeze(-1) + self.dre_solver.B @ ut).squeeze()

        return states, controls

if __name__ == '__main__':
    # Define problem parameters
    n, m = 2, 1  # Dimensions of state and control
    T = 1.0  # Time horizon
    dt = 0.01  # Time step size

    A = torch.tensor([[0.0, 1.0], [-0.1, -0.5]])
    B = torch.tensor([[0.0], [1.0]])
    Q = torch.eye(n)
    R = torch.eye(m)
    initial_state = torch.tensor([[1.0], [0.0]])
    time_steps = int(T / dt)
    signals = torch.sin(torch.linspace(0, T, time_steps).unsqueeze(-1).repeat(1, n))  # Example signal

    # Solve the optimal trading strategy
    trading_strategy = OptimalTradingStrategy(A, B, Q, R, T, dt, initial_state, signals)
    states, controls = trading_strategy.compute_optimal_strategy()

    # Print results
    print("Optimal States:\n", states)
    print("Optimal Controls:\n", controls)