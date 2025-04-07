"""
This script implements other tools one can experiment with when using the martingale test framework.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import defaultdict

from src.adapted_mtgl.utils import get_params
from src.adapted_mtgl.mtgl_test.multiD import basic
from src.adapted_mtgl.mtgl_test.mtgl import mtgl_proj


# (i). Implement the adapted empirical measure from https://arxiv.org/pdf/2002.07261
# do the convergence plot of the MPDs
# uniform on interval (-1/2,1/2) = X, Z; Y = X+Z, X,Z iid
def generate_samples(N, d, T, seed = 42):
    """Generate N i.i.d. samples X^n = (X^n_1, ..., X^n_T) in [0,1]^d."""
    np.random.seed(seed)
    return np.random.rand(N, T, d)

def compute_partition_grid(N, d, T):
    """Compute partition grid centers for adapted empirical measure."""
    if N <= 1:
        raise ValueError("N must be greater than 1 to compute partition grid.")
    r = (T + 1) ** -1 if d == 1 else (d * T) ** -1
    step_size = N ** -r
    grid_1d = np.arange(0, 1 + step_size, step_size)
    grid_centers_1d = (grid_1d[:-1] + grid_1d[1:]) / 2
    if d == 1:
        return grid_centers_1d.reshape(-1, 1)
    else:
        # Create a Cartesian product grid in d-dimensions
        grids = np.meshgrid(*([grid_centers_1d] * d), indexing='ij')
        grid_centers = np.stack(grids, axis=-1).reshape(-1, d)
        return grid_centers
    
def map_to_grid_flat(samples, grid_centers):
    """Map flat (N, d) samples to their nearest grid center."""
    tree = KDTree(grid_centers)
    mapped = np.array([grid_centers[tree.query(x)[1]] for x in samples])
    return mapped

def map_to_grid(samples, grid_centers):
    """Map samples to their nearest grid center."""
    N, T, d = samples.shape
    tree = KDTree(grid_centers)
    mapped_samples = np.zeros_like(samples)
    for n in range(N):
        for t in range(T):
            _, idx = tree.query(samples[n, t])
            mapped_samples[n, t] = grid_centers[idx]
    return mapped_samples

def compute_adapted_empirical_measure(mapped_samples):
    """Compute the adapted empirical measure as a probability mass function."""
    flat_samples = mapped_samples.reshape(mapped_samples.shape[0], -1)
    unique, counts = np.unique(flat_samples, axis=0, return_counts=True)
    probabilities = counts / mapped_samples.shape[0]  # Normalize by N
    return unique, probabilities

def plot_measures(samples, mapped_samples, unique, probabilities):
    """Plot original vs adapted empirical measure."""
    d = samples.shape[2]
    plt.figure(figsize=(10, 4))

    # Original empirical measure
    plt.subplot(1, 2, 1)
    if d == 1:
        plt.hist(samples.reshape(-1), bins=10, alpha=0.7, color='blue', density=True, label='Empirical')
        plt.xlabel("Sample Values")
    else:
        for i in range(d):
            plt.hist(samples[..., i].flatten(), bins=10, alpha=0.5, density=True, label=f'Dim {i+1}')
        plt.xlabel("Component Values")
    plt.ylabel("Density")
    plt.title("Empirical Measure")

    # Adapted empirical measure
    plt.subplot(1, 2, 2)
    if d == 1:
        plt.bar(unique.flatten(), probabilities, width=0.05, color='red', alpha=0.7, label='Adapted')
        plt.xlabel("Grid Centers")
    else:
        plt.scatter(unique[:, 0], unique[:, 1], s=100 * probabilities, c='red', alpha=0.7, label='Adapted')
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
    plt.title("Adapted Empirical Measure")

    plt.legend()
    plt.show()


# def compute_adapted_mpd(X, Y, gamma=1):
#     """
#     Compute MPD(P, gamma) = 2^{1 - gamma} E[ ||X - E[Y | X]||^gamma ]
#     using the adapted empirical measure. X, Y are (n, d) arrays.
#     """
#     n, d = X.shape
#     T = 1
#     grid_centers = compute_partition_grid(n, d, T)
#     mapped_X = map_to_grid_flat(X, grid_centers)
#     XY = np.concatenate([mapped_X, Y], axis=1)
#     grouped = defaultdict(list)
#     for i in range(n):
#         key = tuple(mapped_X[i])
#         grouped[key].append(Y[i])
#     cond_expectation = {k: np.mean(v, axis=0) for k, v in grouped.items()}
#     dist_gamma = [np.linalg.norm(X[i] - cond_expectation[tuple(mapped_X[i])]) ** gamma for i in range(n)]
#     return 2 ** (1 - gamma) * np.mean(dist_gamma)

def compute_adapted_mpd(X, Y, gamma=1):
    """
    Compute MPD(P, gamma) = 2^{1 - gamma} E[ ||X - E[Y | X]||^gamma ]
    using the adapted empirical measure. X, Y are (n, d) arrays.
    This version treats (X, Y) as a T=2 sequence but conditions only on adapted X.
    """
    n, d = X.shape
    T = 2
    sequence = np.stack([X, Y], axis=1)  # shape (n, 2, d)
    grid_centers = compute_partition_grid(n, d, T)
    mapped_sequence = map_to_grid(sequence, grid_centers)  # shape (n, 2, d)
    mapped_X = mapped_sequence[:, 0]  # only use adapted X for conditioning

    grouped = defaultdict(list)
    for i in range(n):
        key = tuple(mapped_X[i])
        grouped[key].append(Y[i])

    cond_expectation = {k: np.mean(v, axis=0) for k, v in grouped.items()}
    dist_gamma = [np.linalg.norm(X[i] - cond_expectation[tuple(mapped_X[i])]) ** gamma for i in range(n)]
    return 2 ** (1 - gamma) * np.mean(dist_gamma)


def plot_adapted_mpd_convergence(num_ls, gamma=1, seed=42):
    mpd_vals = []
    for n in tqdm(num_ls, desc="Computing MPD"):
        X, Y = basic(n_samples=n, seed=seed)
        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
        mpd = compute_adapted_mpd(X, Y, gamma=gamma)
        mpd_vals.append(mpd)

    plt.figure(figsize=(6, 4))
    plt.plot(num_ls, mpd_vals, marker='o')
    plt.xlabel("Number of Samples")
    plt.ylabel("MPD Value")
    plt.title("Convergence of Adapted MPD vs Sample Size")
    plt.grid(True)
    plt.show()

def plot_mpd_convergence_comparison(num_ls, rho=5, sigma=1, lbd=-50, ubd=50, gamma=1, seed=42):
    """
    Compare convergence of adapted MPD vs smoothed MPD across sample sizes.
    """
    mpd_vals = []
    smoothed_mpd_vals = []

    for n in tqdm(num_ls, desc="Computing MPD"):
        X, Y = basic(n_samples=n, seed=seed)
        params = get_params(rho, X, Y, sigma)
        smoothed_mpd = mtgl_proj(params, lbd, ubd)

        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
        mpd = compute_adapted_mpd(X, Y, gamma=gamma)

        mpd_vals.append(mpd)
        smoothed_mpd_vals.append(smoothed_mpd)

    plt.figure(figsize=(6, 4))
    plt.plot(num_ls, mpd_vals, marker='o', label='Adapted MPD')
    plt.plot(num_ls, smoothed_mpd_vals, marker='^', label='Smoothed MPD')
    plt.xlabel("Number of Samples")
    plt.ylabel("MPD Value")
    plt.title("Convergence of Adapted MPD vs Smoothed MPD")
    plt.legend()
    plt.grid(True)
    plt.show()


# if __name__ == '__main__':
#     # d, T, N = 2, 2, 27
#     # samples = generate_samples(N, d, T)
#     # grid_centers = compute_partition_grid(N, d, T)
#     # print(grid_centers)
#     # mapped_samples = map_to_grid(samples, grid_centers)
#     # unique, probabilities = compute_adapted_empirical_measure(mapped_samples)
#     # plot_measures(samples, mapped_samples, unique, probabilities)
#     #num_ls = np.logspace(start=1, stop=6, num=6, base=10, dtype=int)
#     num_ls = np.arange(10,1000+1,100)
#     print(num_ls)
#     plot_adapted_mpd_convergence(num_ls)
