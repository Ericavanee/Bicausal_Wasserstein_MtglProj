"""
Stores object parameters.
"""

import sys
sys.path.append('.')
from dataclasses import dataclass, field
import numpy as np

@dataclass
class HestonParams:
    x0: float = field(default=1.0, metadata={"help": "Initial stock price at time 0."})
    r: float = field(default=0.025, metadata={"help": "Risk-free rate (annualized)."})
    V0: float = field(default=0.04, metadata={"help": "Initial variance at time 0."})
    kappa: float = field(default=0.78, metadata={"help": "Mean reversion speed of variance."})
    mu: float = field(default=0.11, metadata={"help": "Long-term variance."})
    eta: float = field(default=0.68, metadata={"help": "Volatility of variance (vol of vol)."})
    rho: float = field(default=0.044, metadata={"help": "Correlation between stock and variance."})
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

@dataclass
class AsympParams:
    n: int = field(default=100, metadata={"help": "Total number of grid points."})
    d: int = field(default=1, metadata={"help": "Dimension of test."})
    n_sim: int = field(default=100, metadata={"help": "Total number of simulations."})
    domain: list[float] = field(default_factory=lambda: [-50.0, 50.0], metadata={"help": "Domain of integration as [lbd, ubd]."})
    sigma: float = field(default=1.0, metadata={"help": "Standard deviation parameter (must be positive)."})
    rho: float = field(default=5.0, metadata={"help": "Kernel parameter."})
    save: bool = field(default=True, metadata={"help": "Whether to save simulation results."})
    save_dir: str = field(default="sim_data", metadata={"help": "Directory to save simulation results."})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    integration_method: str = field(default='trapez', metadata={"help": "Method of multivariate integration: must be either 'quad', 'trapez,' or 'simps.'"})

    def __post_init__(self):
        if self.d > 1:
            if self.integration_method != 'trapez' and self.integration_method != 'quad' and self.integration_method != 'simps':
                raise ValueError("method must be either 'quad', 'trapez,' or 'simps.'")

@dataclass
class SmoothedMpdParams:
    test_data: str = field(metadata={"help": "Path to .npz file containing both X and Y arrays, each of shape (n, d)."})
    rho: float = field(default=5.0, metadata={"help": "Smoothing kernel parameter."})
    lbd: float = field(default=-50.0, metadata={"help": "Lower bound of integration domain."})
    ubd: float = field(default=50.0, metadata={"help": "Upper bound of integration domain."})
    method: str = field(
        default='mc',
        metadata={
            "help": "Method of calculating the smoothed MPD: either 'mc' for Monte Carlo integration, or 'nquad' for multidimensional integration using nquad."
        }
    )
    n_sim: int = field(
        default=1000,
        metadata={"help": "Number of Monte Carlo simulations (used only if method='mc')."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for Monte Carlo simulation (used only if method='mc')."}
    )
    n_trials: int = field(
        default=1,
        metadata={"help": "Number of trials to average MPD over (for statistical robustness)."}
    )

    def __post_init__(self):
        if self.method == 'mc':
            if self.n_sim <= 0:
                raise ValueError("n_sim must be positive when using Monte Carlo integration.")
        elif self.method != 'nquad':
            raise ValueError("method must be either 'mc' or 'nquad'.")

@dataclass
class AdaptedMpdParams:
    test_data: str = field(metadata={"help": "Path to .npz file containing both X and Y arrays, each of shape (n, d)."})
    method: str = field(default='grid', metadata={"help": "Method for calculating the center: choose 'grid' for using a fixed grid as reference, or 'kmeans' which uses KMeans clustering."})
    gamma: int = field(default=1, metadata={"help": "Gamma parameter (must be greater than or equal to 1)."})
    n_trials: int = field(default=1, metadata={"help": "Number of trials to average MPD over (for statistical robustness)."})

    def __post_init__(self):
        if self.method not in ['grid', 'kmeans']:
            raise ValueError("method must be either 'grid' or 'kmeans'")