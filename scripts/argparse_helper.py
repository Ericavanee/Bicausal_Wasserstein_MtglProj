"""
Stores object parameters.
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
    dt: float = field(default=1/96, metadata={"help": "Time-step size in years."})
    timesteps: int = field(default=96, metadata={"help": "Number of time steps in the simulation."})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})

@dataclass
class NsdeParams:
    device: int = field(default=0, metadata={"help": "Device index for CUDA (0 for first GPU)."})
    n_layers: int = field(default=4, metadata={"help": "Number of layers in the neural SDE network."})
    vNetWidth: int = field(default=50, metadata={"help": "Width of the neural network."})
    experiment: int = field(default=0, metadata={"help": "Experiment index."})
    batch_size: int = field(default=4000, metadata={"help": "Batch size for training."})
    n_epochs: int = field(default=100, metadata={"help": "Number of training epochs."})
    MC_samples_test: int = field(default=1000, metadata={"help": "Number of Monte Carlo test samples."})
    save_dir: str = field(default="data/", metadata={"help": "Directory to save stock price trajectories."})

