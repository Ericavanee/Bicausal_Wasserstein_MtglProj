"""
Script for systematically calling the price_payoff_coupling method of the HestonModel class.
"""

import sys
sys.path.append('.')
import numpy as np
import os
import warnings
import pickle as pkl
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from argparse_helper import HestonParams
from applications.nsde_calibration.heston_call import HestonModel
from src.adapted_mtgl.utils import *

warnings.filterwarnings("ignore")  # Suppress warnings

@dataclass
class CouplingArguments:
    x_path: str = field(default="data/nsde_calibration/stock_traj_LSV.txt", metadata={"help": "Path to the calibrated stock trajectory."})
    v_path: str = field(default="data/nsde_calibration/var_traj_LSV.txt", metadata={"help": "Path to the calibrated stock variance trajectory."})
    maturities: range = field(default=range(16, 33, 16), metadata={"help": "Maturities as a range object. Note maturities here is adjusted by the number of timesteps. For instance, if maturity = 0.5 (i.e. 6 months), and timesteps = 96, then the maturity is 0.5 * 96 = 48."})
    strikes_call: np.ndarray = field(default_factory=lambda: np.arange(0.8, 1.21, 0.02),
                                     metadata={"help": "Array of strike prices for call options."})
    outer_itr: int = field(default=1000, metadata={"help": "Number of outer Monte Carlo paths."})
    inner_itr: int = field(default=50, metadata={"help": "Number of inner Monte Carlo paths."})
    save_path: str = field(default="data/nsde_calibration/coupling_results.pkl", metadata={"help": "Path to save computed results."})

if __name__ == "__main__":
    parser = HfArgumentParser([CouplingArguments, HestonParams])
    args, heston_params = parser.parse_args_into_dataclasses()

    # Convert string inputs to lists
    #strikes_call = np.array([float(k) for k in args.strikes_call.split(",")])
    #maturities = np.array([float(m) for m in args.maturities.split(",")])
    maturities = np.array(list(args.maturities))

    # Initialize Heston model
    heston = HestonModel(
        x0=heston_params.x0,
        r=heston_params.r,
        V0=heston_params.V0,
        kappa=heston_params.kappa,
        mu=heston_params.mu,
        eta=heston_params.eta,
        rho=heston_params.rho,
        dt=heston_params.dt,
        timesteps=heston_params.timesteps,
        seed=heston_params.seed
    )

    print(f"Running price_payoff_coupling with {args.outer_itr} outer paths and {args.inner_itr} inner paths...")

    # Load stock and variance trajectories
    traj_mat = load_txt_as_matrix(args.x_path)
    print(f'Loading calibrated stock trajectory: data shape: {traj_mat.shape}')
    var_mat = load_txt_as_matrix(args.v_path)
    print(f'Loading calibrated variance trajectory: data shape: {var_mat.shape}')

    # Run price_payoff_coupling
    mat_ls, vanilla_payoff_ls = heston.price_payoff_coupling(
        x = traj_mat,
        v = var_mat,
        strikes_call=args.strikes_call,
        maturities=maturities,
        inner_itr=args.inner_itr
    )

    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save results
    with open(args.save_path, 'wb') as f:
        pkl.dump({"mat_ls": mat_ls, "vanilla_payoff_ls": vanilla_payoff_ls}, f)

    print(f"Results saved to {args.save_path}.")

    print("\nPrice-Payoff Coupling for First Maturity and Strike (Outer Path 0, First Time Step):")
    print(mat_ls[0][0, 0, 0], " vs. ", vanilla_payoff_ls[0][0, 0])

# Run the script with the following command:
"""
python scripts/heston_coupling.py \
    -- x_path "data/nsde_calibration/stock_traj_LSV.txt" \
    --v_path "data/nsde_calibration/var_traj_LSV.txt" \
    --maturities "16,32,48" \
    --strikes_call "0.75,0.85,1.0,1.15,1.3" \
    --outer_itr 1000 \
    --inner_itr 50 \
    --save_path "data/nsde_calibration/coupling_results.pkl"
""" 