"""
Script for running asymptotic integral distribution of the smoothed empirical MPD (SE-MPD).
"""


import sys
sys.path.append('.')
import numpy as np
import os
import warnings
import pickle as pkl
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from argparse_helper import AsympParams
from src.adapted_mtgl.mtgl_test.multiD import simulate_integral_distribution, simulate_integral_multi
from src.adapted_mtgl.utils import *

warnings.filterwarnings("ignore")  # Suppress warnings

if __name__ == "__main__":
    parser = HfArgumentParser(AsympParams)
    args, = parser.parse_args_into_dataclasses()

    if args.d == 1:
        intgl_ls = simulate_integral_distribution(args.n,args.n_sim, args.domain, args.sigma, args.rho, args.save, args.save_dir, args.seed)
    else:
        intgl_ls = simulate_integral_multi(args.domain, args.n, args.d, args.sigma, args.rho, args.n_sim, args.integration_method, args.save, args.save_dir, args.seed)

# Example script execution
    """
    python scripts/run_asymptotics.py \
    --d 2 \
    --n 10 \
    --save False
    """

    
