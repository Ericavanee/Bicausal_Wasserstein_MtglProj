"""
Stores object parameters.
"""

import sys
sys.path.append('.')
from dataclasses import dataclass, field
import numpy as np

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
    batch_size: int = field(default=1000, metadata={"help": "Number of Monte Carlo trajectories generated in each stochastic gradient descent (SGD) iteration."}) # 40000 for original (number of training samples)
    n_epochs: int = field(default=100, metadata={"help": "Number of training epochs."}) # 1000 for original
    MC_samples_test: int = field(default=5000, metadata={"help": "Number of Brownian motion paths for final evaluation."}) # 200000 for original (number of testing samples)
    save_dir: str = field(default="data/nsde_calibration/", metadata={"help": "Directory to save stock price trajectories."})

@dataclass
class OptionParam:
    maturities: range = field(default=range(16, 33, 16), metadata={"help": "Maturities as a range object. Note maturities here is adjusted by the number of timesteps. For instance, if maturity = 0.5 (i.e. 6 months), and timesteps = 96, then the maturity is 0.5 * 96 = 48."})
    timesteps: int = field(default=96, metadata={"help": "Number of time steps in the simulation."})
    strikes_call: np.ndarray = field(default_factory=lambda: np.arange(0.8, 1.21, 0.02),
                                     metadata={"help": "Array of strike prices for call options."})
    stock_init: float = field(default=1.0, metadata={"help": "Initial stock price."})
    rate: float = field(default=0.025, metadata={"help": "Risk-free rate."})

