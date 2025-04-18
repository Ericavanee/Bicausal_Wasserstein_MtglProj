"""
This script implements functions for option pricing and loss evaluation based on the Heston model. 
It includes martingale projection-based loss functions for calibration.
"""

import numpy as np
from src.adapted_mtgl.mtgl_test.multiD import smoothing_function



def mtgLoss_vanilla(rho, calibrated_payoff, market_price):
    """
    Computes the martingale projection loss for vanilla options.

    Parameters:
    - rho (float): Kernel parameter for loss computation.
    - calibrated_payoff (list of ndarrays): Simulated vanilla option payoffs (grouped by maturity).
    - market_price (list of ndarrays): Market-observed vanilla option prices.

    Returns:
    - avg_loss (float): Average squared loss over all maturities.
    """

    num_maturities = len(calibrated_payoff)  # Number of maturities
    num_iter, num_strikes = calibrated_payoff[0].shape  # Number of simulations and strikes

    total_loss = 0  # Accumulate loss

    for maturity_idx in range(num_maturities):
        loss_per_iter = []
        for j in range(num_iter):
            # Compute difference between calibrated and market payoffs
            diff = calibrated_payoff[maturity_idx][j].T.reshape(num_strikes, 1) - market_price[maturity_idx][j]
            kernel_weighted_diff = np.multiply(diff, smoothing_function(rho, 1, diff))
            loss_per_iter.append(np.sum(kernel_weighted_diff**2))

        total_loss += sum(loss_per_iter)

    avg_loss = total_loss / (num_maturities * num_iter)
    return avg_loss


def mtgLoss_pair(rho, calibrated_payoff, market_price, summation=False):
    """
    Computes the martingale projection loss for both vanilla and exotic options.

    Parameters:
    - rho (float): Kernel parameter for loss computation.
    - calibrated_payoff (tuple): Tuple containing (exotic_payoff, vanilla_payoff).
    - market_price (tuple): Tuple containing (exotic_market_price, vanilla_market_price).
    - summation (bool): If True, returns total loss; otherwise, returns separate losses.

    Returns:
    - If summation=True, returns total loss (float).
    - If summation=False, returns tuple (exotic_loss, vanilla_loss).
    """

    num_maturities = len(calibrated_payoff[1])  # Number of maturities
    vanilla_loss = 0  # Initialize vanilla option loss

    # Compute vanilla option loss
    for maturity_idx in range(num_maturities):
        diff = calibrated_payoff[1][maturity_idx] - market_price[1][maturity_idx]
        kernel_weighted_diff = np.multiply(diff, smoothing_function(rho, 1, diff))
        vanilla_loss += np.sum(kernel_weighted_diff**2)

    # Compute exotic option loss
    exotic_diff = calibrated_payoff[0] - market_price[0]
    exotic_kernel_weighted_diff = np.multiply(exotic_diff, smoothing_function(rho, 1, exotic_diff))
    exotic_loss = np.sum(exotic_kernel_weighted_diff**2)

    if summation:
        total_loss = exotic_loss + vanilla_loss
        print("Total loss of vanilla and exotic options:", total_loss)
        return total_loss
    else:
        print("Loss in vanilla options:", vanilla_loss)
        print("Loss in exotic options:", exotic_loss)
        return exotic_loss, vanilla_loss
