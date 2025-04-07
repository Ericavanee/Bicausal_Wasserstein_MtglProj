"""
Implements Multi-dimensional Martingale test for gamma = 1.
Includes the simulation of the asymptotic distribution of the martingale test test satistic.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.integrate import quad
from tqdm import tqdm
from src.adapted_mtgl.mtgl_test.multiD import smoothing_function
from src.adapted_mtgl.utils import get_params

warnings.filterwarnings("ignore")

# Part 1: helper functions

def cutoff(result, conf, display = False):
    percentile_cutoff = np.percentile(result, conf)
    if display:
        print(f"cutoff value: {percentile_cutoff}")
    return percentile_cutoff

def get_bounds(domain, d):
    lbd, ubd = domain
    bounds = [(lbd, ubd) for _ in range(d)]
    return bounds

# Part 2: mtgl test
# Kernel is not the problem; integrand mismatch - quad supplying a point and kernel needs to be calculated point-wise. 

def integrand(a, params):
    """
    a: tuple of floats of length d (from nquad or MC)
    params: dictionary with keys 'rho', 'x', 'y', 'sigma'
    """
    rho = params['rho']
    x = params['x']
    y = params['y']
    if x.ndim == 1:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
    a = np.array(a)  # shape (d,)
    n = len(x)
    sum_vec = np.zeros_like(x[0])  # shape (d,) — initialize vector sum

    for i in range(n):
        diff = a - x[i]  # shape (d,)
        k_val = smoothing_function(rho, 1, diff)  # scalar
        sum_vec += (y[i] - x[i]) * k_val  # shape (d,)

    return np.linalg.norm(sum_vec) / n


def mtgl_proj(params, lbd, ubd):
    warnings.simplefilter("ignore")
    x = params['x']
    d = x.shape[1] if x.ndim > 1 else 1

    if d == 1:
        f = lambda a: integrand(a, params)
        I = quad(f, lbd, ubd)
    else:
        bounds = get_bounds([lbd, ubd], d)
        f = lambda *args: integrand(args, params)
        I = nquad(f, bounds)

    return I[0]


def mtgl_test(params, lbd, ubd, conf, result, display = True): # nquad version
    bond = cutoff(result, conf)
    x = params['x']
    n = len(x)
    proj = np.sqrt(n) * mtgl_proj(params, lbd, ubd)
    if proj <= bond:
        if display:
            print(f"Accept null hypothesis with {conf}% confidence.")
        return proj, True
    else:
        if display:
            print(f"Reject null hypothesis with {conf}% confidence.")
        return proj, False

    

# Monte Carlo projection estimate to imporvie high dimensional integration complexity
def mtgl_proj_mc(params, lbd, ubd, n_samples=1000, seed=0):
    np.random.seed(seed)
    x = params['x']
    d = x.shape[1] if x.ndim > 1 else 1
    volume = (ubd - lbd) ** d

    a_samples = np.random.uniform(lbd, ubd, size=(n_samples, d))

    integrand_vals = np.array([
        integrand(a, params) for a in tqdm(a_samples, desc="Evaluating integrand...")
    ])

    mc_estimate = volume * np.mean(integrand_vals)
    return mc_estimate


def mtgl_test_mc(params, lbd, ubd, conf, result, n_samples=1000, seed=0, display = True):
    bond = cutoff(result, conf)
    x = params['x']
    n = len(x)
    proj = np.sqrt(n) * mtgl_proj_mc(params, lbd, ubd, n_samples, seed)
    if proj <= bond:
        if display:
            print(f"Accept null hypothesis with {conf}% confidence.")
        return proj, True
    else:
        if display:
            print(f"Reject null hypothesis with {conf}% confidence.")
        return proj, False


def asymp_rej_rate_mc(n_sim, params, lbd, ubd, conf, result, n_samples=1000): # explore impact of different seeds in monte carlo integration
    """
    Estimates asymptotic rejection rate of the MC martingale test.
    """
    ls = []
    for i in range(n_sim):
        _, decision = mtgl_test_mc(params, lbd, ubd, conf, result, n_samples, seed = i)
        if not decision:
            ls.append(1)
    return sum(ls) / n_sim

def power_curve_mc(grid, n_sim, params, lbd, ubd, conf, result, n_samples=1000): # explore impact of different seeds in monte carlo integration
    """
    Plots power curve over a perturbation grid using MC version of the test.
    """
    rej_rate = []
    rho = params['rho']
    x = params['x']
    y = params['y']
    
    for delta in grid:
        y_shifted = y + delta
        new_params = get_params(rho, x, y_shifted)
        rate = asymp_rej_rate_mc(n_sim, new_params, lbd, ubd, conf, result, n_samples)
        rej_rate.append(rate)

    plt.plot(grid, rej_rate)
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Rejection rate")
    plt.title("Power Curve (MC Version)")
    plt.grid(True)
    plt.show()

    return grid, rej_rate


# def power_curve_nquad(grid, params, lbd, ubd, conf, result):
#     """
#     Plots power curve over a perturbation grid using `nquad` version of the test.
#     """
#     rej_rate = []
#     rho = params['rho']
#     x = params['x']
#     y = params['y']

#     for delta in grid:
#         y_shifted = y + delta
#         new_params = get_params(rho, x, y_shifted)
#         _, decision = mtgl_test(new_params, lbd, ubd, conf, result)
#         rate = 1 if not decision else 0
#         rej_rate.append(rate)

#     plt.plot(grid, rej_rate)
#     plt.xlabel("Shift / Perturbation magnitude")
#     plt.ylabel("Rejection rate")
#     plt.title("Power Curve (nquad Version)")
#     plt.grid(True)
#     plt.show()

#     return grid, rej_rate

    