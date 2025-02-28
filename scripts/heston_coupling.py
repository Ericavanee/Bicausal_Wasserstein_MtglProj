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

warnings.filterwarnings("ignore")  # Suppress warnings

@dataclass
class CouplingArguments:
    strikes_call: str = field(metadata={"help": "Comma-separated list of strike prices."})
    maturities: str = field(metadata={"help": "Comma-separated list of maturities in years."})
    outer_itr: int = field(default=100, metadata={"help": "Number of outer Monte Carlo paths."})
    inner_itr: int = field(default=500, metadata={"help": "Number of inner Monte Carlo paths."})
    save_path: str = field(default="results.pkl", metadata={"help": "Path to save computed results."})

if __name__ == "__main__":
    parser = HfArgumentParser([CouplingArguments, HestonParams])
    args, heston_params = parser.parse_args_into_dataclasses()

    # Convert string inputs to lists
    strikes_call = np.array([float(k) for k in args.strikes_call.split(",")])
    maturities = np.array([float(m) for m in args.maturities.split(",")])

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

    # Run price_payoff_coupling
    mat_ls, vanilla_payoff_ls = heston.price_payoff_coupling(
        strikes_call=strikes_call,
        maturities=maturities,
        outer_itr=args.outer_itr,
        inner_itr=args.inner_itr
    )

    # Save results
    with open(args.save_path, 'wb') as f:
        pkl.dump({"mat_ls": mat_ls, "vanilla_payoff_ls": vanilla_payoff_ls}, f)

    print(f"Results saved to {args.save_path}.")
