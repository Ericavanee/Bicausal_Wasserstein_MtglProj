"""
This script implements stock price simulations based on the Heston model and computes prices for vanilla and exotic options using Monte Carlo methods.
"""

import numpy as np
import matplotlib.pyplot as plt


def Heston_stock(
    x0=1, r=0.025, V0=0.04, kappa=0.78, mu=0.11, eta=0.68, rho=0.044, 
    itr=int(1e7), dt=1/96, timesteps=96
):
    """
    Simulates stock price trajectories using the Heston stochastic volatility model.

    Parameters:
    - x0 (float): Initial stock price.
    - r (float): Risk-free rate.
    - V0 (float): Initial variance.
    - kappa (float): Mean-reversion rate of variance.
    - mu (float): Long-term mean variance.
    - eta (float): Volatility of variance process.
    - rho (float): Correlation between stock price and variance.
    - itr (int): Number of Monte Carlo simulations.
    - dt (float): Time step size.
    - timesteps (int): Number of time steps.

    Returns:
    - X_t (ndarray): Stock price trajectories of shape (itr, timesteps).
    - running_max (ndarray): Running maximum of stock prices for exotic options.
    """

    # Initialize stock price and variance arrays
    X_t = np.zeros((timesteps, itr))
    V_t = np.zeros((timesteps, itr))
    V_t[0, :] = V0
    X_t[0, :] = x0
    running_max = X_t[0, :]

    # Generate correlated Gaussian noise
    cov_matrix = [[1, rho], [rho, 1]]
    Z = np.random.multivariate_normal([0, 0], cov_matrix, (itr, timesteps)).T
    Z1, Z2 = Z[0], Z[1]

    # Simulate stock price and variance over time
    for i in range(1, timesteps):
        # Variance process update
        V_t[i, :] = np.maximum(
            V_t[i - 1, :] + kappa * (mu - V_t[i - 1, :]) * dt + eta * np.sqrt(V_t[i - 1, :] * dt) * Z2[i, :],
            0
        )

        # Stock price update
        X_t[i, :] = X_t[i - 1, :] + r * X_t[i - 1, :] * dt + np.sqrt(V_t[i, :] * dt) * X_t[i - 1, :] * Z1[i, :]

        # Update running maximum
        running_max = np.maximum(running_max, X_t[i, :])

    return X_t.T, running_max


def plot_heston(stock_paths, time_grid):
    """
    Plots stock price trajectories simulated from the Heston model.

    Parameters:
    - stock_paths (ndarray): Array of shape (n_paths, timesteps) containing stock trajectories.
    - time_grid (ndarray): Time points corresponding to timesteps.
    """
    for path in stock_paths:
        plt.plot(time_grid, path)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Heston Model Stock Price Trajectories")
    plt.show()


def price_option(strikes_call, maturities, x0=1, r=0.025, V0=0.04, kappa=0.78, 
                 mu=0.11, eta=0.68, rho=0.044, itr=int(1e7), dt=1/96, timesteps=96):
    """
    Prices vanilla and exotic options using Monte Carlo simulations under the Heston model.

    Parameters:
    - strikes_call (list): Strike prices for vanilla options.
    - maturities (list): Maturity time steps.
    - x0, r, V0, kappa, mu, eta, rho: Heston model parameters.
    - itr (int): Number of Monte Carlo simulations.
    - dt (float): Time step size.
    - timesteps (int): Number of time steps.

    Returns:
    - exotic_option_price (ndarray): Exotic option price estimates over time.
    - vanilla_option_prices (list of ndarrays): Vanilla option prices at different maturities.
    """

    # Create time grid
    timegrid = np.linspace(0, 1, timesteps + 1)
    last_maturity = maturities[-1]

    # Generate Monte Carlo stock paths
    outer_stock, running_max = Heston_stock(x0, r, V0, kappa, mu, eta, rho, itr, dt, timesteps)

    # Initialize storage for vanilla and exotic options
    vanilla_option_prices = [np.zeros((itr, len(strikes_call), maturity)) for maturity in maturities]
    exotic_option_price = np.zeros((itr, last_maturity))

    for i in range(itr):
        running_max_copy = running_max
        traj = outer_stock[i]

        for j in range(last_maturity):
            # Generate new Monte Carlo paths from the current stock price
            if j == 0:
                inner_stock, inner_max = outer_stock, running_max
            else:
                inner_stock, inner_max = Heston_stock(x0, r, V0, kappa, mu, eta, rho, itr, dt, timesteps=last_maturity - j)

            # Compute vanilla option prices at different maturities
            for k, maturity in enumerate(maturities):
                if j < maturity:
                    S_T = inner_stock.T[last_maturity - j - 1]  # Stock value at maturity
                    for idx, strike in enumerate(strikes_call):
                        vanilla_price = np.exp(-r * maturity) * np.clip(S_T - strike, 0, None)
                        vanilla_option_prices[k][i, idx, j] = vanilla_price.mean()

            # Update running max for exotic options
            running_max_copy = np.maximum(running_max_copy, inner_max)
            exotic_option_price[i, j] = np.exp(-r * timegrid[last_maturity]) * (running_max_copy - S_T).mean()

    # Compute expected option prices across Monte Carlo trials
    exotic_option_price = exotic_option_price.mean(axis=0)
    vanilla_option_prices = [np.mean(price, axis=0) for price in vanilla_option_prices]

    return exotic_option_price, vanilla_option_prices
