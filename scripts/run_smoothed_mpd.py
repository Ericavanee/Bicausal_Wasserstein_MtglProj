"""
Script for running smoothed empirical MPD (SE-MPD).
"""

import sys
sys.path.append('.')
import numpy as np
import warnings
from transformers.hf_argparser import HfArgumentParser
from argparse_helper import SmoothedMpdParams
from src.adapted_mtgl.mtgl_test.mtgl import mtgl_proj, mtgl_proj_mc
from src.adapted_mtgl.utils import get_params
from src.adapted_mtgl.mtgl_test.mtgl_couplings import basic

warnings.filterwarnings("ignore")  # Suppress warnings

if __name__ == "__main__":
    parser = HfArgumentParser(SmoothedMpdParams)
    args, = parser.parse_args_into_dataclasses()

    if not args.test_data:
        print("No filenames given.")
        print("Default to running SE-MPD for basic gaussian martingale couplings of sample size 1000 in d=1.")
        X,Y = basic(1000)
    else:
        loaded = np.load(args.test_data)
        X = loaded['X']
        Y = loaded['Y']

    params = get_params(args.rho, X, Y)

    if args.method == 'mc':
        result = mtgl_proj_mc(params,args.lbd,args.ubd,args.n_sim,args.seed)
    elif args.method == 'nquad':
        result = mtgl_proj(params, args.lbd, args.ubd)
    else:
        raise(Exception("Method must be either 'mc' for monte carlo integration, or 'nquad', for nquad multivariate integration."))
    
    print(f"The Smoothed Emprical Martingale Projection Distance for the given testing couplings is {result}.")

    """
    python scripts/run_smoothed_mpd.py \
    --test_data ""
    """



