"""
Script for systematically running the adapted LV-based Neural SDE model.
"""

import sys
sys.path.append('..')
import os
import warnings
import pickle as pkl
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from argparse_helper import NsdeParams, OptionParam
from applications.nsde_calibration.adapted_LV import *
warnings.filterwarnings("ignore")  # Suppress warnings

if __name__ == "__main__":
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

    # Initialize LV-based Neural SDE model
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
    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_dir)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    error_path = os.path.join(args.save_dir, "error_hedge_LV.txt")
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
    best_model = train_nsde(lv_model, z_test, CONFIG)

    # Generate stock price trajectory and diffusion parameters
    batch_size = args.batch_size
    period_length = 16
    batch_z = torch.randn(batch_size, n_steps, device=device)
    T = max(maturities)
    stock_path, var_path, diffusion, _, _, _, _, _, _ = best_model(S0, batch_z, batch_size, T, period_length)

    # Save stock price trajectory and diffusion parameters
    stock_save_path = os.path.join(args.save_dir, "stock_traj_LV.txt")
    diffusion_save_path = os.path.join(args.save_dir, "diffusion_LV.txt")
    var_save_path = os.path.join(args.save_dir, "var_traj_LV.txt")

    np.savetxt(stock_save_path, stock_path.cpu().numpy())
    print(f"Saving stock price trajectory to file {stock_save_path}...")
    np.savetxt(diffusion_save_path, diffusion.cpu().numpy())
    print(f"Saving diffusion parameters to file {diffusion_save_path}...")
    np.savetxt(var_save_path, var_path.cpu().numpy())
    print(f"Saving stock variance trajectory to file {var_save_path}...")

    print("Run completed successfully.")


# Example script execution
    """
    python scripts/run_adapted_LV.py \
    --device 0 \
    --n_layers 4 \
    --vNetWidth 128 \
    --MC_samples_test 1000 \
    --batch_size 64 \
    --n_epochs 100 \
    --save_dir data/nsde_calibration/
    """
