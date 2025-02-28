"""
This script implements stock price simulations based on the Heston model and computes prices for vanilla and exotic call options using a nested Monte Carlo methods.
"""

import numpy as np
import matplotlib.pyplot as plt

def Heston_stock(
    x0=1.0,
    r=0.025,
    V0=0.04,
    kappa=0.78,
    mu=0.11,
    eta=0.68,
    rho=0.044,
    itr=10000,
    dt=1/96,
    timesteps=96
):
    """
    Simulates stock price (and variance) trajectories from t=0 to t=1 under the Heston model using a simple Euler scheme, and also tracks the pathwise
    running maximum for exotic options pricing

    Parameters
    ----------
    x0 : float
        Initial stock price at time 0.
    r : float
        Risk-free rate (annualized).
    V0 : float
        Initial variance at time 0.
    kappa, mu, eta, rho : floats
        Heston model parameters (mean reversion speed, long-term variance,
        vol of vol, correlation).
    itr : int
        Number of Monte Carlo paths (outer or single-run).
    dt : float
        Time-step in years. dt=1/96 => 96 steps per year.
    timesteps : int
        Number of discrete steps => total times = timesteps+1 from 0..1.

    Returns
    -------
    stock_paths : ndarray, shape (itr, timesteps+1)
        stock_paths[i, j] is the stock price for path i at step j.
    var_paths   : ndarray, shape (itr, timesteps+1)
        var_paths[i, j] is the variance for path i at step j.
    running_max : ndarray, shape (itr, timesteps+1)
        running_max[i, j] is the maximum of stock price for path i
        up to time j.

    Notes
    -----
    - We generate correlated noise (Z1, Z2) with correlation rho.
    - Euler updates:
        V_{n+1} = max( V_n + kappa*(mu - V_n)*dt + eta*sqrt(V_n*dt)*Z2, 0 )
        S_{n+1} = S_n + r*S_n*dt + sqrt(V_{n+1}*dt)*S_n*Z1
      for n=0...timesteps-1.
    """
    n_times = timesteps + 1  # total number of discrete points (0..timesteps)

    # Allocate arrays
    stock_paths = np.zeros((itr, n_times))
    var_paths   = np.zeros((itr, n_times))
    running_max = np.zeros((itr, n_times))

    # Initial conditions
    stock_paths[:, 0] = x0
    var_paths[:, 0]   = V0
    running_max[:, 0] = x0

    # Correlated Gaussian draws: shape => (itr, timesteps, 2)
    cov = [[1.0, rho], [rho, 1.0]]
    Z = np.random.multivariate_normal([0, 0], cov, (itr, timesteps))

    # Euler-Maruyama updates over time steps
    for t in range(1, n_times):
        Z1 = Z[:, t-1, 0]
        Z2 = Z[:, t-1, 1]

        # Update variance
        var_paths[:, t] = np.maximum(
            var_paths[:, t-1]
            + kappa*(mu - var_paths[:, t-1]) * dt
            + eta * np.sqrt(np.clip(var_paths[:, t-1], 0, None) * dt) * Z2,
            0.0
        )

        # Update stock price
        stock_paths[:, t] = (
            stock_paths[:, t-1]
            + r * stock_paths[:, t-1] * dt
            + np.sqrt(var_paths[:, t] * dt) * stock_paths[:, t-1] * Z1
        )

        # Update running max
        running_max[:, t] = np.maximum(running_max[:, t-1], stock_paths[:, t])

    return stock_paths, var_paths, running_max


def Heston_stock_from_state(
    S0,
    V0,
    r,
    kappa,
    mu,
    eta,
    rho,
    itr=1000,
    dt=1/96,
    timesteps=48
):
    """
    Generates new Heston paths starting from (S0, V0) for each path up to 'timesteps' steps, using the same Euler scheme.

    Parameters
    ----------
    S0 : float
        Initial stock price (for each path).
    V0 : float
        Initial variance (for each path).
    r, kappa, mu, eta, rho : floats
        Heston model parameters.
    itr : int
        Number of new inner paths to simulate.
    dt : float
        Time-step in years.
    timesteps : int
        Number of steps for the inner simulation.

    Returns
    -------
    stock_paths_inner : ndarray, shape (itr, timesteps+1)
        Paths from t_j up to t_j + timesteps*dt.
        stock_paths_inner[i, 0] = S0
    var_paths_inner   : ndarray, shape (itr, timesteps+1)
        The variance along each inner path.
    """
    n_times = timesteps + 1
    stock_paths_inner = np.zeros((itr, n_times))
    var_paths_inner   = np.zeros((itr, n_times))

    # Set the initial state for all inner paths
    stock_paths_inner[:, 0] = S0
    var_paths_inner[:, 0]   = V0

    cov = [[1.0, rho], [rho, 1.0]]
    Z = np.random.multivariate_normal([0,0], cov, (itr, timesteps))

    for t in range(1, n_times):
        Z1 = Z[:, t-1, 0]
        Z2 = Z[:, t-1, 1]

        var_paths_inner[:, t] = np.maximum(
            var_paths_inner[:, t-1]
            + kappa*(mu - var_paths_inner[:, t-1])*dt
            + eta*np.sqrt(np.clip(var_paths_inner[:, t-1],0,None)*dt)*Z2,
            0.0
        )

        stock_paths_inner[:, t] = (
            stock_paths_inner[:, t-1]
            + r*stock_paths_inner[:, t-1]*dt
            + np.sqrt(var_paths_inner[:, t]*dt)*stock_paths_inner[:, t-1]*Z1
        )

    return stock_paths_inner, var_paths_inner


def plot_heston(stock_paths, time_grid):
    """
    Plots stock price trajectories.

    Parameters
    ----------
    stock_paths : ndarray, shape (n_paths, n_times)
    time_grid   : ndarray, shape (n_times,)

    Notes
    -----
    - For large n_paths, consider plotting fewer samples or using alpha to reduce clutter.
    """
    for i in range(stock_paths.shape[0]):
        plt.plot(time_grid, stock_paths[i, :], alpha=0.6)
    plt.xlabel("Time (years)")
    plt.ylabel("Stock Price")
    plt.title("Sample Heston Trajectories")
    plt.show()


def price_option_nested_mc(
    strikes_call,
    T_in_years=1.0,     # final maturity
    t_in_years=0.5,     # intermediate time for the option price
    outer_itr=1000,
    inner_itr=2000,
    dt=1/96,
    total_timesteps=96,
    x0=1.0,
    r=0.025,
    V0=0.04,
    kappa=0.78,
    mu=0.11,
    eta=0.68,
    rho=0.044
):
    """
    Nested MC for an intermediate time t_in_years under Heston, 
    returning both vanilla calls (for multiple strikes) and 
    a single exotic payoff of type: max(S up to T) - S(T).

    1) Outer MC: up to t_in_years => get (S_j^(i), V_j^(i)).
    2) Inner MC: from that state to T => compute payoffs, discount them from T to t_in_years.
    3) Average over outer paths => unconditional price at time t_in_years.

    The exotic payoff is computed on the *inner* path as:
        exotic_payoff = max_{t_in_years <= u <= T} S_u - S_T,
    clipped at 0.
    Discount = exp(-r*(T - t_in_years)).

    Returns
    -------
    call_prices_tj : ndarray, shape (n_strikes,)
        The time-t_in_years price for each strike in strikes_call.
    exotic_price_tj : float
        The time-t_in_years price of the exotic (lookback-type) payoff.
    """
    # Create the time grid
    timegrid = np.linspace(0, 1, total_timesteps+1)

    # Indices for t_in_years and T_in_years
    j_idx = np.searchsorted(timegrid, t_in_years)
    j_idx = min(j_idx, total_timesteps)
    T_idx = np.searchsorted(timegrid, T_in_years)
    T_idx = min(T_idx, total_timesteps)

    if T_in_years < t_in_years:
        raise ValueError("Final maturity T_in_years must be >= t_in_years.")

    # 1) Outer MC from 0..t_in_years
    #    We'll only simulate 'j_idx' steps if t_in_years < 1
    outer_stock, outer_var, _outer_max = Heston_stock(
        x0=x0, r=r, V0=V0, kappa=kappa, mu=mu, eta=eta, rho=rho,
        itr=outer_itr,
        dt=dt,
        timesteps=j_idx
    )
    # shapes => (outer_itr, j_idx+1)

    # The stock and variance at time t_in_years
    S_j = outer_stock[:, j_idx]   # shape=(outer_itr,)
    V_j = outer_var[:, j_idx]

    n_strikes = len(strikes_call)
    sum_conditional_calls = np.zeros(n_strikes)
    sum_conditional_exotic = 0.0

    # # of steps from t_in_years..T_in_years
    inner_steps = T_idx - j_idx
    if inner_steps < 0:
        raise ValueError("T_in_years < t_in_years, not valid. (Should be caught above)")

    # discount factor from t_in_years..T_in_years
    discount = np.exp(-r*(T_in_years - t_in_years))

    # 2) Loop over outer paths
    for i in range(outer_itr):
        # current outer path's state
        S0_inner = S_j[i]
        V0_inner = V_j[i]

        # Inner simulation from t_in_years..T_in_years
        stock_inner, var_inner, running_max_inner = Heston_stock_from_state(
            S0=S0_inner,
            V0=V0_inner,
            r=r, kappa=kappa, mu=mu, eta=eta, rho=rho,
            itr=inner_itr,
            dt=dt,
            timesteps=inner_steps
        )
        # shapes => (inner_itr, inner_steps+1)

        # final index in the inner path = inner_steps => corresponds to T_in_years
        S_T_inner = stock_inner[:, inner_steps]
        S_max_T_inner = running_max_inner[:, inner_steps]

        # 2a) Vanilla calls for each strike
        for k_idx, K in enumerate(strikes_call):
            payoff_call = np.clip(S_T_inner - K, 0, None)
            sum_conditional_calls[k_idx] += discount * payoff_call.mean()

        # 2b) Exotic payoff: max(S up to T) - S(T), from t_in_years..T_in_years
        payoff_exotic = np.clip(S_max_T_inner - S_T_inner, 0, None)
        sum_conditional_exotic += discount * payoff_exotic.mean()

    # 3) Average across outer paths
    vanilla_prices_tj = sum_conditional_calls / outer_itr
    exotic_price_tj = sum_conditional_exotic / outer_itr

    return vanilla_prices_tj, exotic_price_tj


# -------------------------- Example Usage --------------------------------
# if __name__ == "__main__":
#     # Example: We want to price at t_in_years=0.5, with final T_in_years=1.0
#     # for multiple strikes in [0.8..1.2].
#     strikes = np.linspace(0.8, 1.2, 5)
#     # We'll do outer_itr=50, inner_itr=200, just for demonstration
#     prices_nested = price_option_nested_mc(
#         strikes_call=strikes,
#         T_in_years=1.0,
#         t_in_years=0.5,
#         outer_itr=50,
#         inner_itr=200,
#         dt=1/96,
#         total_timesteps=96,
#         x0=1.0,
#         r=0.025,
#         V0=0.04,
#         kappa=0.78,
#         mu=0.11,
#         eta=0.68,
#         rho=0.044
#     )
#     print("Nested MC call prices at t=0.5, for T=1.0:\nStrikes:", strikes)
#     print("Prices:", prices_nested)
#
#     # We can also plot a few of the outer paths for demonstration:
#     # outer_stock, _ = Heston_stock(..., itr=5, timesteps=48) # 5 outer paths
#     # timegrid_outer = np.linspace(0, 0.5, 48+1)
#     # plot_heston(outer_stock, timegrid_outer)
# -------------------------------------------------------------------------
