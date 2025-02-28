"""
Stores previous functions.
"""

import numpy as np
def calibratedOption_payoff(strikes_call, maturities, theta_ls, itr, r=0.025, dt=1/96, timesteps=96):
    """
    Computes the option payoffs (both vanilla and exotic) using Monte Carlo simulations with calibrated parameters.

    Parameters:
    - strikes_call (list): List of strike prices for vanilla options.
    - maturities (list): List of maturity time steps.
    - theta_ls (list of tuples): List of Heston model parameters for different maturities: [x0, r, V0, kappa, mu, eta, rho].
    - itr (int): Number of Monte Carlo simulations.
    - dt (float): Time step size.
    - timesteps (int): Number of time steps.

    Returns:
    - exotic_option_price (ndarray): Mean exotic option prices over time.
    - vanilla_option_prices (list of ndarrays): Mean vanilla option prices at different maturities.
    """

    # Create time grid and get last maturity
    timegrid = np.linspace(0, 1, timesteps + 1)
    last_maturity = maturities[-1]

    # Initialize holders for vanilla options at different maturities
    vanilla_option_prices = [np.zeros((itr, len(strikes_call), maturity)) for maturity in maturities]

    # Initialize holder for exotic options
    exotic_option_price = np.zeros((itr, last_maturity))

    # Loop over maturities to compute option payoffs
    for maturity_idx, maturity in enumerate(maturities):
        # Generate stock paths using Heston model with the corresponding parameters
        stock, _, running_max = Heston_stock(*theta_ls[maturity_idx], itr=itr, dt=dt, timesteps=timesteps)

        for j in range(itr):
            # Create a copy of the running max
            running_max_copy = running_max
            traj = stock[j]

            for t in range(maturity):
                # Generate inner Monte Carlo trials if t > 0
                if t == 0:
                    inner_stock, inner_max = stock, running_max
                else:
                    inner_stock, inner_max = Heston_stock(*theta_ls[maturity_idx], itr, dt, timesteps=maturity - t)

                # Get stock price at maturity
                S_T = inner_stock.T[maturity - t - 1]

                # Compute vanilla option prices
                for idx, strike in enumerate(strikes_call):
                    vanilla_price = np.exp(-r * (maturity - t)) * np.clip(S_T - strike, 0, None)
                    vanilla_option_prices[maturity_idx][j, idx, t] = vanilla_price.mean()

                # Compute exotic option price only for the last maturity
                if maturity == last_maturity:
                    running_max_copy = np.maximum(running_max_copy, inner_max)
                    exotic_option_price[j, t] = np.exp(-r * timegrid[last_maturity]) * (running_max_copy - S_T).mean()

    # Compute expected option prices across Monte Carlo trials
    exotic_option_price = exotic_option_price.mean(axis=0)
    vanilla_option_prices = [np.mean(price, axis=0) for price in vanilla_option_prices]

    return exotic_option_price, vanilla_option_prices