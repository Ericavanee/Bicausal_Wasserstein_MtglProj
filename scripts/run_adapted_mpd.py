"""
Script for running adapted empirical MPD.
"""

import sys
sys.path.append('.')
import numpy as np
import warnings
from transformers.hf_argparser import HfArgumentParser
from argparse_helper import AdaptedMpdParams
from src.adapted_mtgl.mtgl_test.ada_emp_dist import compute_adapted_mpd
from src.adapted_mtgl.mtgl_test.mtgl_couplings import generate_uniform_martingale_coupling

warnings.filterwarnings("ignore")  # Suppress warnings

if __name__ == "__main__":
    parser = HfArgumentParser(AdaptedMpdParams)
    args, = parser.parse_args_into_dataclasses()

    if not args.test_data:
        print("No filenames given.")
        print("Default to running SE-MPD for basic uniform martingale couplings (~Unif[-0.5,0.5]) of sample size 1000 in d=1.")
        X,Y = generate_uniform_martingale_coupling(1000)
    else:
        loaded = np.load(args.test_data)
        X = loaded['X']
        Y = loaded['Y']

    result = compute_adapted_mpd(X, Y, args.method,args.gamma)
    
    print(f"The Adapted Emprical Martingale Projection Distance for the given testing couplings is {result}.")

    """
    python scripts/run_adapted_mpd.py \
    --test_data ""
    """



