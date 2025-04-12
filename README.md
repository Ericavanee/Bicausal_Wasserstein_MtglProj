# Bicausal Wasserstein Martingale Projection
The *bicausal* or *adapted* Wasserstein martingale projection is a novel method for multi-dimensional martingale tests on data couplings. It relies on the *smoothed empirical martingale projection distance* as its test statistic, which achieves a dimension-independent convergence rate of $O(N^{1/2})$ in the sample size $N$. To the best of our knowledge, it is the first **consistent** martingality test for high-dimensional regimes.

🔗 Paper link: [Empirical martingale projections via the adapted Wasserstein distance](https://arxiv.org/pdf/2401.12197)

# Abstract
Given a collection of multidimensional pairs $\{(X_i,Y_i)\}_{1 \leq i\leq n}$, we study the problem of projecting the associated suitably smoothed empirical measure onto the space of martingale couplings (i.e. distributions satisfying $\mathbb{E}[Y|X]=X$) using the adapted Wasserstein distance. We call the resulting distance the *smoothed empirical martingale projection distance* (SE-MPD), for which we obtain an explicit characterization. We also show that the space of martingale couplings remains invariant under the smoothing operation. We study the asymptotic limit of the SE-MPD, which converges at a parametric rate as the sample size increases if the pairs are either i.i.d.~or satisfy appropriate mixing assumptions. Additional finite-sample results are also investigated. Using these results, we introduce a novel consistent martingale coupling hypothesis test, which we apply to test the existence of arbitrage opportunities in recently introduced neural network-based generative models for asset pricing calibration.

<p align="center">
  <img src="documentation/loglog1d.png" width="48%" />
  <img src="documentation/loglog3d.png" width="48%" />
</p>

<p align="center">
  <em>Figure:</em> Log-log convergence comparison between the adapted MPD (based on the 
  <a href="https://arxiv.org/abs/2002.07261" target="_blank"><em>adapted empirical measure</em></a>) 
  and the smoothed MPD (<em>ours</em>) for <em>d = 1</em> (left) and <em>d = 3</em> (right), using Uniform[-0.5,0.5] increments.
</p>

# Repo Structure
- **`scripts`**: `python` scripts for getting calibrated stock and variance trajectory using neural-SDE as followed by the pipeline developed by [Gierjatowicz et. al.](https://arxiv.org/abs/2007.04154) and getting couplings of **actual** call options payoff calculated via a nested MC conditioning on the stock price $S_t$ at each intermediate time $t$ and the
payoff estimated by coniditioning on $S_0$.

    The primary scripts are as follows:   
    - **`run_adapted_LV.py`**: Run `adapted_LV.py` to get stock and variance trajectory calibrated via the LV neural-SDE model.
    - **`run_adapted_LSV.py`**: Run `adapted_LV.py` to get stock and variance trajectory calibrated via the LSV neural-SDE model.
    - **`heston_coupling.py`**: Run `adapted_LV.py` to get couplings of **actual** call options payoff calculated via a nested MC conditioning on the stock price $S_t$ at each intermediate time $t$ and the payoff estimated by coniditioning on $S_0$ using a given calibrated stock trajectory.
    - **`run_asymptotics.py`**: Run `run_asymptotics.py` to simulate the SE-MPD test statistic distribution. This is then used to produce the cutoff values one uses to conduct the martingale test.
    - **`run_adapted_mpd.py`**, **`run_smoothed_mpd.py`**: Run `run_adapted_mpd.py`, `run_smoothed_mpd.py` respectively to calculate the adapted empirical MPD and the SE-MPD of a given test couplings dataset.
- **`demos`**: `jupyter` tutorials for functionalities implemented in the repo. It includes demos for running the applications (such as `nsde_calibration.ipynb` for options pricing calibration) and various simulations (see `simulation.ipynb`).
    - **`simulation.ipynb`** explores the effect of $\sigma$ in the SE-MPD test statistic.
    - **`synthetic_experiment.ipynb`** gives examples of conducting the multidimensional martingale coupling test using SE-MPD.
    - **`adapted_empirical_distance.ipynb`** explores the convergence property of SE-MPD in juxtaposition to the adapted empirical MPD.
    - **`nsde_calibration.ipynb`** gives examples on how one calibrates vanilla European option prices using the apartus given by [Gierjatowicz et. al.](https://arxiv.org/abs/2007.04154) and how one obtains the (market_price, payoff) couplings for the martingale coupling test to validate the efficacy of the option calibration procedure. 
    - **`markov.ipynb`** gives examples of conducting mutlidimensional SE-MPD based martingale test on markov chain couplings.
- **`src`**: `python` source scripts for running the martingale test. We highlight a few items.
    - **`src/adapted_mtgl/mtgl_test/simulated_data`** contains pre-simulated SE-MPD test statistic distribution for given parameters $d, \rho, \sigma$. One can directly use this to conduct the martingale coupling test without having to resimulate.
    - **`src/adapted_mtgl/mtgl_test/multiD.py`** contains source codes for simulating the SE-MPD test statistic distribution. For mutivariate integration required to calculate the SE-MPD statistic, we support two methos:
        - `nquad` which is a numerical solver supported by `scipy`.
        - `mc` which supports monte carlo simulation to approximate the integral. This scales much more effectively and is to be preferred to `nquad` especially in high dimensions. 
    - **`src/adapted_mtgl/mtgl_test/mtgl.py`** contains the source function for implementing the SE-MPD martingale coupling test.
    - **`src/adapted_mtgl/mtgl_test/ada_emp_dist.py`** implements the adapted empirical measure and the corresponding adapted empirical MPD. It also implements functionalities that compare the convergence behavior between the adapted empirical MPD and our SE-MPD. For the adapted empirical measure, we support two methods of calculation:
        - `grid`: this method follows the original instructions in Definition 1.2 of paper [Estimating processes in adapted Wasserstein distance](https://arxiv.org/abs/2002.07261) using a predefined fixed grid to define the centers. Empirical samples are then grouped to these centers via nearest-neighbor assignment using a `KDTree`.
        -  `kmeans`: this method is forked from the implementation of the adapted empirical measure in Github repo ["aotnumerics"](https://github.com/stephaneckstein/aotnumerics), which uses k-means clustering from `sklearn` to define the centers dynamically. 
- **`data`**: cotains the call prices used in the neural SDE calibration example as well as the calibratoed stock trajectories for both Local Stochastic Volatility Model (LSV) and the Loal Volatility (LV) model. One can also calibrate these themselves by running `scripts/run_adapted_LSV.py` and/or `scripts/run_adapted_LV.py`.
- **`applications`**: contains source codes implementing the two applications discussed in Section 4.3 of our paper: markov and nsde. Tutorials to run both are contained in the folder `demos`.

# Citation

If you find this repository helpful in your research or applications, please consider citing our paper:

```bibtex
@misc{blanchet2024empiricalmartingaleprojectionsadapted,
  title={Empirical martingale projections via the adapted Wasserstein distance}, 
  author={Jose Blanchet and Johannes Wiesel and Erica Zhang and Zhenyuan Zhang},
  year={2024},
  eprint={2401.12197},
  archivePrefix={arXiv},
  primaryClass={math.PR},
  url={https://arxiv.org/abs/2401.12197}
}

