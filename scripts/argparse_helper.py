"""
Stores Heston model parameters.
"""

import sys
sys.path.append('.')
from dataclasses import dataclass, field

@dataclass
class HestonParams:
    x0: float = field(default=100.0, metadata={"help": "Initial stock price at time 0."})
    r: float = field(default=0.02, metadata={"help": "Risk-free rate (annualized)."})
    V0: float = field(default=0.04, metadata={"help": "Initial variance at time 0."})
    kappa: float = field(default=0.5, metadata={"help": "Mean reversion speed of variance."})
    mu: float = field(default=0.04, metadata={"help": "Long-term variance."})
    eta: float = field(default=0.3, metadata={"help": "Volatility of variance (vol of vol)."})
    rho: float = field(default=-0.5, metadata={"help": "Correlation between stock and variance."})
    dt: float = field(default=1/252, metadata={"help": "Time-step size in years."})
    timesteps: int = field(default=252, metadata={"help": "Number of time steps in the simulation."})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
