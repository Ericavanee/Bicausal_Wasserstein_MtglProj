"""
Script for systematically running the adapted LV-based Neural SDE model.
"""

import sys
sys.path.append('.')
import os
import warnings
import pickle as pkl
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from argparse_helper import NsdeParams
from applications.nsde_calibration.adapted_LV import *
warnings.filterwarnings("ignore")  # Suppress warnings

if __name__ == "__main__":
    parser = HfArgumentParser([NsdeParams])
    args = parser.parse_args_into_dataclasses()[0]

    # Set device
    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
        torch.cuda.set_device(args.device)
    else:
        device = "cpu"

    # Load market prices
    data = torch.load("Call_prices_59.pt")
    print("Loading data from Call_prices_59.pt.")
    print(f"Data Shape: {data.shape}")

    # Set up training - Strike values, time discretization, and maturities
    strikes_call = np.arange(0.8, 1.21, 0.02)
    n_steps = 96
    timegrid = torch.linspace(0, 1, n_steps + 1).to(device)
    maturities = range(16, 65, 16)
    n_maturities = len(maturities)

    # Initialize LV-based Neural SDE model
    S0 = 1
    rate = 0.025
    lv_model = Net_LV(
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
    lv_model.to(device)
    lv_model.apply(init_weights)

    # Monte Carlo test data
    MC_samples_test = args.MC_samples_test
    z_test = torch.randn(MC_samples_test, n_steps, device=device)
    z_test = torch.cat([z_test, -z_test], 0)  # Antithetic Brownian paths

    # Logging file
    with open("error_hedge.txt", "w") as f:
        f.write("epoch,error_hedge_2,error_hedge_inf\n")

    # Training Configuration
    CONFIG = {
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "maturities": maturities,
        "n_maturities": n_maturities,
        "strikes_call": strikes_call,
        "timegrid": timegrid,
        "n_steps": n_steps,
        "target_data": data
    }

    # Train NSDE
    best_model = train_nsde(lv_model, z_test, CONFIG)

    # Generate stock price trajectory and diffusion parameters
    batch_size = args.batch_size
    period_length = 16
    batch_z = torch.randn(batch_size, n_steps, device=device)
    T = max(maturities)
    stock_path, var_path, diffusion, _, _, _, _, _ = best_model(S0, batch_z, batch_size, T, period_length)

    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save stock price trajectory and diffusion parameters
    stock_save_path = "data/stock_traj_LV.txt"
    diffusion_save_path = "data/LV_diffusion.txt"
    var_save_path = "data/var_traj_LV.txt"

    print(f"Saving stock price trajectory to {stock_save_path} and diffusion parameters to {diffusion_save_path}.")
    np.savetxt(stock_save_path, stock_path.cpu().numpy())
    np.savetxt(diffusion_save_path, diffusion.cpu().numpy())
    np.savetxt(var_save_path, var_path.cpu().numpy())

    print("Run completed successfully.")
