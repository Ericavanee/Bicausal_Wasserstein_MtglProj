"""
This script implements stock price simulations based on the Heston model and computes prices for vanilla and exotic call options using a nested Monte Carlo methods.
"""

import numpy as np
import matplotlib.pyplot as plt

class HestonModel:
    """
    A class for simulating stock price dynamics under the Heston model and pricing vanilla and exotic options using Monte Carlo methods.
    """
    def __init__(self, x0=1.0, r=0.025, V0=0.04, kappa=0.78, mu=0.11, eta=0.68, rho=0.044, dt=1/96, timesteps=96, seed = 42):
        """
        Initializes the Heston model parameters.

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
        seed : int
            Random seed for reproducibility.
        """
        self.x0 = x0
        self.r = r
        self.V0 = V0
        self.kappa = kappa
        self.mu = mu
        self.eta = eta
        self.rho = rho
        self.dt = dt
        self.timesteps = timesteps
        self.n_times = timesteps + 1
        self.seed = seed

    def Heston_stock(self, itr=1000):
        """
        Simulates stock price (and variance) trajectories from t=0 to t=1 under the Heston model using a simple Euler scheme, and also tracks the pathwise 
        running maximum for exotic options pricing.

        Parameters
        ----------
        itr : int
        Number of Monte Carlo paths (outer or single-run).

        """
        np.random.seed(self.seed)
        stock_paths = np.zeros((itr, self.n_times))
        var_paths = np.zeros((itr, self.n_times))
        running_max = np.zeros((itr, self.n_times))

        stock_paths[:, 0] = self.x0
        var_paths[:, 0] = self.V0
        running_max[:, 0] = self.x0

        cov = [[1.0, self.rho], [self.rho, 1.0]]
        Z = np.random.multivariate_normal([0, 0], cov, (itr, self.timesteps))

        for t in range(1, self.n_times):
            Z1, Z2 = Z[:, t-1, 0], Z[:, t-1, 1]
            var_paths[:, t] = np.maximum(
                var_paths[:, t-1] + self.kappa * (self.mu - var_paths[:, t-1]) * self.dt +
                self.eta * np.sqrt(np.clip(var_paths[:, t-1], 0, None) * self.dt) * Z2,
                0.0
            )
            stock_paths[:, t] = stock_paths[:, t-1] + self.r * stock_paths[:, t-1] * self.dt + np.sqrt(var_paths[:, t] * self.dt) * stock_paths[:, t-1] * Z1
            running_max[:, t] = np.maximum(running_max[:, t-1], stock_paths[:, t])

        return stock_paths, var_paths, running_max
    
    def Heston_stock_from_state(self, S0, V0, itr=1000, timesteps=48):
        """
        Generates new Heston paths (stock price and variance) starting from (S0, V0) 
        for each path up to 'timesteps' steps, using the same Euler scheme.
        Also tracks the running maximum of the stock over these inner paths.

        Parameters
        ----------
        S0 : float
            Initial stock price (for each path).
        V0 : float
            Initial variance (for each path).
        itr : int, default=1000
            Number of new inner paths to simulate.
        timesteps : int, default=48
            Number of steps for the inner simulation.

        Returns
        -------
        stock_paths_inner : ndarray, shape (itr, timesteps+1)
            Stock price paths from t_j up to t_j + timesteps*dt.
            stock_paths_inner[i, 0] = S0 for each path i.
        var_paths_inner : ndarray, shape (itr, timesteps+1)
            The variance along each inner path.
        running_max_inner : ndarray, shape (itr, timesteps+1)
            running_max_inner[i, t] is the maximum stock price observed 
            on the i-th path up to (and including) time t of the inner simulation.

        Notes
        -----
        - For step t in [1..timesteps], we do:
            var_paths_inner[i, t] = max(
            var_paths_inner[i, t-1] + kappa*(mu - var_paths_inner[i, t-1])*dt
            + eta * sqrt(var_paths_inner[i, t-1]*dt)*Z2[i],
            0
            )

            stock_paths_inner[i, t] = stock_paths_inner[i, t-1]
                                    + r*stock_paths_inner[i, t-1]*dt
                                    + sqrt(var_paths_inner[i, t]*dt)*stock_paths_inner[i, t-1]*Z1[i]

            running_max_inner[i, t] = max(running_max_inner[i, t-1], stock_paths_inner[i, t])
        """
        np.random.seed(self.seed)
        stock_paths_inner = np.zeros((itr, timesteps + 1))
        var_paths_inner = np.zeros((itr, timesteps + 1))
        running_max_inner = np.zeros((itr, timesteps + 1))

        stock_paths_inner[:, 0] = S0
        var_paths_inner[:, 0] = V0
        running_max_inner[:, 0] = S0

        cov = [[1.0, self.rho], [self.rho, 1.0]]
        Z = np.random.multivariate_normal([0, 0], cov, (itr, timesteps))

        for t in range(1, timesteps + 1):
            Z1, Z2 = Z[:, t-1, 0], Z[:, t-1, 1]
            var_paths_inner[:, t] = np.maximum(
                var_paths_inner[:, t-1] + self.kappa * (self.mu - var_paths_inner[:, t-1]) * self.dt +
                self.eta * np.sqrt(np.clip(var_paths_inner[:, t-1], 0, None) * self.dt) * Z2,
                0.0
            )
            stock_paths_inner[:, t] = stock_paths_inner[:, t-1] + self.r * stock_paths_inner[:, t-1] * self.dt + np.sqrt(var_paths_inner[:, t] * self.dt) * stock_paths_inner[:, t-1] * Z1
            running_max_inner[:, t] = np.maximum(running_max_inner[:, t-1], stock_paths_inner[:, t])

        return stock_paths_inner, var_paths_inner, running_max_inner
    
    def plot_paths(self, stock_paths):
        """
        Plots stock price trajectories.

        Parameters
        ----------
        stock_paths : ndarray, shape (n_paths, n_times)

        Notes
        -----
        - For large n_paths, consider plotting fewer samples or using alpha to reduce clutter.
        """
        time_grid = np.linspace(0, 1, self.n_times)
        for i in range(stock_paths.shape[0]):
            plt.plot(time_grid, stock_paths[i, :], alpha=0.6)
        plt.xlabel("Time (years)")
        plt.ylabel("Stock Price")
        plt.title("Sample Heston Trajectories")
        plt.show()

    def price_option_for_t(self, strikes_call, maturities_in_years, t_in_years=0.5, outer_itr=1000, inner_itr=2000):
        """
        Price option for an intermediate time t_in_years using a nested MC from time t_in_years to multiple final maturities in maturities_in_years under the Heston model. 
        We'll only simulate up to t_in_years for the outer pass.
        Returns:
          1) A 2D array of vanilla call prices, shape (n_maturities, n_strikes).
          2) A 1D array of exotic prices, shape (n_maturities,).
        
        Steps:
          1) Outer MC from 0..t_in_years (only once). 
             - We'll store S_j(i), V_j(i) for each outer path i at time t_in_years.
          2) For each maturity T in 'maturities_in_years':
             - For each outer path i, run an inner MC from (S_j(i), V_j(i))..T
             - Compute call payoffs (S_T - K)_+ for each strike, and the exotic payoff 
               max_{t_in_years..T}(S_u) - S_T.
             - Discount from T back to t_in_years with exp(-r*(T - t_in_years)).
             - Average over inner, then sum up across outer. 
          3) Divide by outer_itr => get final price at t_in_years for each (K, T).
        
        Parameters
        ----------
        strikes_call : 1D array-like of length n_strikes
        maturities_in_years : 1D array-like of length n_maturities
            Each T >= t_in_years. Note that this is a list or array of final maturities with T_i >= t_in_years
        t_in_years : float
            The intermediate time at which we want the option prices. 
            Must be <= each T in 'maturities_in_years'.
        outer_itr, inner_itr : int
            The number of outer and inner paths for the nested MC.

        Returns
        -------
        vanilla_prices : ndarray, shape (n_maturities, n_strikes)
            vanilla_prices[m, k] = time-t_in_years price of call with strike 
            strikes_call[k], final maturity = maturities_in_years[m].
        exotic_prices : ndarray, shape (n_maturities,)
            exotic_prices[m] = time-t_in_years price of the exotic payoff 
            max_{t_in_years..T}(S_u) - S_T for maturity = maturities_in_years[m].

        Notes
        -----
        - We do a single outer pass up to t_in_years, then for each T we do outer_itr 
        inner simulations (i.e. outer_itr * inner_itr total).
        - The final discount for maturity T is exp(-r*(T - t_in_years)).
        - If you have many maturities, this can be computationally expensive, 
        as you do a new set of inner sims for each T * each outer path i.
        
        """
        np.random.seed(self.seed)
        # Sort maturities in ascending order (just in case user gave them unsorted)
        maturities_in_years = np.array(maturities_in_years, dtype=float)
        maturities_in_years.sort()

        # Ensure all T >= t_in_years
        if any(m < t_in_years for m in maturities_in_years):
            raise ValueError("All maturities must be >= t_in_years")

        # Create the base time grid 0..1
        timegrid = np.linspace(0, 1, self.timesteps+1)

        # Find index j_idx for t_in_years
        j_idx = np.searchsorted(timegrid, t_in_years)
        j_idx = min(j_idx, self.timesteps)

        # 1) Outer MC from 0..t_in_years
        outer_stock, outer_var, _ = self.Heston_stock(itr=outer_itr)
        # shapes => (outer_itr, j_idx+1)

        # State at time t_in_years for each outer path
        S_j = outer_stock[:, j_idx]
        V_j = outer_var[:, j_idx]

        n_strikes = len(strikes_call)
        n_maturities = len(maturities_in_years)

        # Prepare output arrays
        vanilla_prices = np.zeros((n_maturities, n_strikes))
        exotic_prices  = np.zeros(n_maturities)

        # 2) For each maturity T in 'maturities_in_years'
        for m_idx, T in enumerate(maturities_in_years):
            # Find how many steps from t_in_years..T
            T_idx = np.searchsorted(timegrid, T)
            T_idx = min(T_idx, self.timesteps)

            inner_steps = T_idx - j_idx  # might be 0 if T == t_in_years
            if inner_steps < 0:
                raise ValueError(f"Maturity {T} < t_in_years {t_in_years}, not valid")

            discount = np.exp(-self.r * (T - t_in_years))

            sum_conditional_calls = np.zeros(n_strikes)
            sum_conditional_exotic = 0.0

            # For each outer path i, do an inner pass
            for i in range(outer_itr):
                # Start from S_j[i], V_j[i]
                stock_inner, var_inner, max_inner = self.Heston_stock_from_state(
                    S0 = S_j[i],
                    V0 = V_j[i],
                    itr=inner_itr,
                    timesteps=inner_steps
                )
                # final index = inner_steps => time T
                S_T_inner = stock_inner[:, inner_steps]
                S_max_T_inner = max_inner[:, inner_steps]

                # Vanilla calls
                for k_idx, strike in enumerate(strikes_call):
                    payoff_call = np.clip(S_T_inner - strike, 0, None)
                    sum_conditional_calls[k_idx] += discount * payoff_call.mean()

                # Exotic payoff: max_{t_in_years..T}(S_u) - S_T
                payoff_exotic = np.clip(S_max_T_inner - S_T_inner, 0, None)
                sum_conditional_exotic += discount * payoff_exotic.mean()

            # Average across outer paths
            vanilla_prices[m_idx, :] = sum_conditional_calls / outer_itr
            exotic_prices[m_idx]     = sum_conditional_exotic / outer_itr

        return vanilla_prices, exotic_prices
    

    def price_payoff_coupling(self, strikes_call, maturities, outer_itr = 10, inner_itr = 10):
        """
        Return coupling of price of vanilla options using nested MC and payoff calculated directly by conditioning on S0.
        This is to be passed to directly conduct the martingale test.

        Parameters
        ----------
        strikes_call : 1D array-like of length n_strikes
        maturities : 1D array-like of length n_maturities
            Each T >= t_in_years. Note that this is a list or array of final maturities with T_i >= t_in_years
        outer_itr, inner_itr : int
            The number of outer and inner paths for the nested MC.
        """
        np.random.seed(self.seed)
        # Sort maturities in ascending order (just in case user gave them unsorted)
        maturities = np.array(maturities, dtype=float)
        maturities.sort()
        
        # generate timegrid given time steps.
        timegrid = np.linspace(0,1,self.timesteps+1)
        
        # record the last maturity as ind_T.
        ind_T = maturities[-1]

        # 1) generate outer MC (calibrated MC)
        outer_stock, outer_var, _ = self.Heston_stock(itr=outer_itr) # shapes => (outer_itr, j_idx+1)

        # initiate holders for vanilla option at different maturities
        # structure: mat_ls[maturity_i_index][outer_itr, len(strikes_call), num_time_steps_up_to_maturity_i]
        mat_ls = []
        for i in range(len(maturities)):
            mat_ls.append(np.zeros((outer_itr,len(strikes_call),maturities[i])))
            

        for i in range(outer_itr):
            traj = outer_stock[i] 
            # iterate through each time step up to the final maturity date ind_T
            for j in range(0, ind_T):
                # generate inner monte carlo trials based on real parameters
                inner_steps = ind_T-j
                # Start from outer_stock[i], outer_var[i]
                inner_stock, _inner_var, _ = self.Heston_stock_from_state(
                    S0 = outer_stock[i],
                    V0 = outer_var[i],
                    itr=inner_itr,
                    timesteps=inner_steps
                )

                # iterate through each maturity
                for k in range(len(maturities)):
                    if j < maturities[k]: 
                        my_mat = maturities[k] # current maturity date T
                        S_T = inner_stock[:, -1] # get S_T for all inner paths, => shape (inner_itr,)
                        for idx, strike in enumerate(strikes_call):
                            price_vanilla = np.exp(-self.r*(my_mat-j))*np.clip(np.array(S_T-strike),0, np.inf) # correct discounting with exp(-r*(T-t)).
                            # => shape (inner_itr,)
                            # for each maturity and at each time step
                            mat_ls[k][i,idx,j] = price_vanilla.mean() # taking the mean of the inner paths.
            
            
        # generate (calibrated) payoffs by directly conditioning on S0.
        vanilla_payoff_ls = []
        for i in range(len(maturities)):
            vanilla_payoff = np.zeros((outer_itr, len(strikes_call)))
            for idx, strike in enumerate(strikes_call):
                vanilla_payoff[:,idx] = np.exp(-self.r*my_mat)*np.clip(outer_stock.T[maturities[i]]-strike,0,np.inf)
            vanilla_payoff_ls.append(vanilla_payoff)

        return mat_ls, vanilla_payoff_ls



if __name__ == "__main__":
    # Instantiate the Heston Model
    heston = HestonModel(
        x0=100,  # Initial stock price
        r=0.02,  # Risk-free rate
        V0=0.04, # Initial variance
        kappa=0.5, mu=0.04, eta=0.3, rho=-0.5, # Heston parameters
        dt=1/252, timesteps=252,  # 252 trading days per year
        seed=42  # Fixing seed for reproducibility
    )

    # Simulate stock price paths
    n_paths = 5  # Number of paths to plot
    stock_paths, var_paths, running_max = heston.Heston_stock(itr=n_paths)

    # Plot the first few simulated stock paths
    heston.plot_paths(stock_paths)

    # Define strike prices and maturities for option pricing
    strike_prices = np.array([90, 100, 110])  # ATM, ITM, OTM
    maturities = np.array([0.5, 1.0])  # 6 months, 1 year

    # Price vanilla and exotic options using nested Monte Carlo
    vanilla_prices, exotic_prices = heston.price_option_for_t(
        strikes_call=strike_prices, 
        maturities_in_years=maturities, 
        t_in_years=0.25,  # Pricing at 3 months in
        outer_itr=500, inner_itr=1000
    )

    # Print the results
    print("Vanilla Call Option Prices (rows: maturities, columns: strikes):")
    print(vanilla_prices)

    print("\nExotic Option Prices (maturities):")
    print(exotic_prices)

    # Perform martingale test using price-payoff coupling
    mat_ls, vanilla_payoff_ls = heston.price_payoff_coupling(
        strikes_call=strike_prices,
        maturities=maturities,
        outer_itr=100,
        inner_itr=200
    )

    # Print one example result
    print("\nPrice-Payoff Coupling for First Maturity and Strike (Outer Path 0, First Time Step):")
    print(mat_ls[0][0, 0, 0], " vs. ", vanilla_payoff_ls[0][0, 0])