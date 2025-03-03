"""
Script for systematically running the adapted LSV-based Neural SDE model.
"""

import sys
sys.path.append('..')
import os
import warnings
import pickle as pkl
import torch
import numpy as np
from transformers.hf_argparser import HfArgumentParser
from argparse_helper import NsdeParams, OptionParam
from applications.nsde_calibration.adapted_LSV import *

warnings.filterwarnings("ignore")  # Suppress warnings

if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser([NsdeParams, OptionParam])
    args, option_params = parser.parse_args_into_dataclasses()

    # Set device
    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
        torch.cuda.set_device(args.device)
    else:
        device = "cpu"

    # Load market prices
    data = torch.load("data/Call_prices_59.pt")
    print("Loading data from data/Call_prices_59.pt.")
    print(f"Data Shape: {data.shape}")

    # Extract option parameters
    strikes_call = option_params.strikes_call
    maturities = option_params.maturities
    n_maturities = len(maturities)
    n_steps = option_params.timesteps
    timegrid = torch.linspace(0, 1, n_steps + 1).to(device)
    S0 = option_params.stock_init
    rate = option_params.rate

    # Initialize LSV-based Neural SDE model
    lsv_model = Net_LSV(
        dim=1,
        timegrid=timegrid,
        strikes_call=strikes_call,
        n_layers=args.n_layers,
        vNetWidth=args.vNetWidth,
        device=device,
        n_maturities=n_maturities,
        maturities=maturities,
        rate=rate
    )
    lsv_model.to(device)
    lsv_model.apply(init_weights)

    # Monte Carlo test data
    MC_samples_test = args.MC_samples_test
    z_test = torch.randn(MC_samples_test, n_steps, device=device)
    z_test = torch.cat([z_test, -z_test], 0)  # Antithetic Brownian paths

    # Logging file
    save_dir = os.path.dirname(args.save_dir)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    error_path = os.path.join(args.save_dir, "error_hedge_LSV.txt")
    with open(error_path, "w") as f:
        f.write("epoch,error_hedge_2,error_hedge_inf\n")

    # Training Configuration
    CONFIG = {
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "device": device,
        "maturities": maturities,
        "n_maturities": n_maturities,
        "strikes_call": strikes_call,
        "timegrid": timegrid,
        "n_steps": n_steps,
        "target_data": data,
        "init_stock": S0,
        "experiment": args.experiment,
        "save_dir": args.save_dir
    }

    # Train NSDE
    best_model = train_nsde(lsv_model, z_test, CONFIG)

    # Generate stock price trajectory
    batch_size = args.batch_size
    period_length = 16
    batch_z = torch.randn(batch_size, n_steps, device=device)
    T = max(maturities)
    path, var_path, _, _, _, _, _, _ = best_model(S0, batch_z, batch_size, T, period_length)

    # Ensure save directory exists
    save_path = os.path.join(args.save_dir, "stock_traj_LSV.txt")
    save_var_path = os.path.join(args.save_dir, "var_traj_LSV.txt")

    # Save stock price trajectory
    np.savetxt(save_path, path.cpu().numpy())
    print(f"Saving stock price trajectory to file {save_path}...")
    np.savetxt(save_var_path, var_path.cpu().numpy())
    print(f"Saving stock variance trajectory to file {save_var_path}...")

    print("Run completed successfully.")

    # Example script execution
    """
    python scripts/run_adapted_LSV.py \
    --device 0 \
    --n_layers 4 \
    --vNetWidth 50 \
    --MC_samples_test 5000 \
    --batch_size 1000 \
    --n_epochs 100 \
    --save_dir data/nsde_calibration/
    """
