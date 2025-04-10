"""
This script implements the adapted empirical measure from https://arxiv.org/pdf/2002.07261
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree # finds nearest neighbor (centers) to group samples
from collections import defaultdict

from src.adapted_mtgl.utils import get_params
from src.adapted_mtgl.mtgl_test.multiD import generate_uniform_martingale_coupling
from src.adapted_mtgl.mtgl_test.mtgl import mtgl_proj, mtgl_proj_mc

# do the convergence plot of the MPDs
# uniform on interval (-1/2,1/2) = X, Z; Y = X+Z, X,Z iid
def generate_samples(N, d, T, seed = 42):
    """Generate N i.i.d. samples X^n = (X^n_1, ..., X^n_T) in [0,1]^d."""
    np.random.seed(seed)
    return np.random.rand(N, T, d)

def compute_partition_grid(N, d, T): # correct
    """Compute partition grid centers for adapted empirical measure."""
    if N <= 1:
        raise ValueError("N must be greater than 1 to compute partition grid.")
    
    r = (T + 1) ** -1 if d == 1 else (d * T) ** -1

    # print(f'r = {r}')
    step_size = N ** -r
    # print(f'step size = {step_size}')
    grid_1d = np.arange(0, 1 + step_size, step_size)
    # print(grid_1d)
    grid_centers_1d = (grid_1d[:-1] + grid_1d[1:]) / 2

    # num_bins = int(np.floor(N ** r))
    # edges = np.linspace(0, 1, num_bins + 1)
    # grid_centers_1d = (edges[:-1] + edges[1:]) / 2

    if d == 1:
        return grid_centers_1d.reshape(-1, 1)
    else:
        grids = np.meshgrid(*([grid_centers_1d] * d), indexing='ij')
        grid_centers = np.stack(grids, axis=-1).reshape(-1, d)
        return grid_centers

def map_to_grid(samples, grid_centers): # correct 
    """Map samples to their nearest grid center."""
    N, T, d = samples.shape
    # print(f'samples shape (N,T,d): {(N,T,d)}')
    # print(f'samples: {samples}')
    # print(f'grid center shape:{grid_centers.shape}')
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

def plot_measures(samples, unique, probabilities):
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
#     Compute MPD(P, gamma) = 2^{1 - gamma} E_{\hat{\mu}}[ ||X - E_{\hat{\mu}}[Y | X]||^gamma ]
#     where the expectation is taken under the adapted empirical measure based on X only.
#     This version supports multi-dimensional X.
#     """
#     n, d = X.shape

#     # Treat X as a T=1 sequence of d-dimensional samples
#     X_seq = X[:, np.newaxis, :]  # shape (n, 1, d)
#     grid_centers = compute_partition_grid(n, d, T=1)
#     mapped_X_seq = map_to_grid(X_seq, grid_centers)  # shape (n, 1, d)
#     mapped_X = mapped_X_seq[:, 0, :]  # shape (n, d)   c1: (Xi,yi)...c2: (xi',yi'), and c1,...,c2 is mapped using only Xi's not (xi,yi) because we take conditional expectation on X

#     grouped = defaultdict(list)
#     for i in range(n):
#         key = tuple(mapped_X[i])
#         grouped[key].append((X[i], Y[i]))

#     mpd_sum = 0.0
#     for group in grouped.values():
#         xs = np.array([x for x, _ in group])
#         ys = np.array([y for _, y in group])
#         print(xs,ys)
#         y_mean = np.mean(ys, axis=0)
#         mpd_sum += np.sum(np.linalg.norm(xs - y_mean, axis=1) ** gamma)

#     return 2 ** (1 - gamma) * mpd_sum / n

def compute_adapted_mpd(X, Y, gamma=1):
    """
    Compute MPD(P, gamma) = 2^{1 - gamma} E_{\hat{\mu}}[ ||\varphi^N(X) - E_{\hat{\mu}}[\varphi^N(Y) | \varphi^N(X)]||^gamma ]
    where the expectation is taken under the adapted empirical measure based on mapped X and mapped Y.
    This version supports multi-dimensional X and Y.
    """
    n, d = X.shape

    # Stack (X, Y) as a T=2 trajectory for joint mapping
    sequence = np.stack([X, Y], axis=1)  # shape (n, 2, d)
    grid_centers = compute_partition_grid(n, d, T=2)
    mapped_sequence = map_to_grid(sequence, grid_centers)  # shape (n, 2, d)

    mapped_X = mapped_sequence[:, 0, :]
    #print(mapped_X)
    mapped_Y = mapped_sequence[:, 1, :]
    #print(mapped_Y)

    # # Map X and Y separately using T=1
    # X_seq = X[:, np.newaxis, :]  # shape (n, 1, d)
    # Y_seq = Y[:, np.newaxis, :]  # shape (n, 1, d)

    # grid_centers = compute_partition_grid(n, d, T=1)
    # mapped_X1 = map_to_grid(X_seq, grid_centers)[:, 0, :]  # shape (n, d)
    # # print(mapped_X)
    # mapped_Y1 = map_to_grid(Y_seq, grid_centers)[:, 0, :]  # shape (n, d)
    # #print(mapped_Y)

    grouped = defaultdict(list)
    for i in range(n):
        key = tuple(mapped_X[i])
        grouped[key].append(mapped_Y[i])

    mpd_sum = 0.0
    for key, y_vals in grouped.items():
        g = np.array(key)
        # print(f'x center: {g}')
        y_mean = np.mean(y_vals, axis=0)
        # print(y_vals)
        dist = np.linalg.norm(g - y_mean) ** gamma
        mpd_sum += len(y_vals) * dist

    return 2 ** (1 - gamma) * mpd_sum / n


def plot_adapted_mpd_convergence(num_ls, d=1, gamma=1, seed=0, mpd_vals = None):
    if not mpd_vals:
        mpd_vals = []
        for n in tqdm(num_ls, desc="Computing MPD"):
            X, Y = generate_uniform_martingale_coupling(n_samples=n, d=d, seed=seed)
            if d==1:
                X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
            mpd = compute_adapted_mpd(X, Y, gamma=gamma)
            mpd_vals.append(mpd)

    # Linear plot
    plt.figure(figsize=(6, 4))
    plt.plot(num_ls, mpd_vals, marker='o')
    plt.xlabel("Number of Samples")
    plt.ylabel("MPD Value")
    plt.title("Convergence of Adapted MPD vs Sample Size")
    plt.grid(True)
    plt.show()

    # Log-log plot
    plt.figure(figsize=(6, 4))
    plt.loglog(num_ls, mpd_vals, marker='o')

    plt.xlabel("log(Number of Samples)")
    plt.ylabel("log(MPD Value)")
    plt.title("Log-Log Convergence of Adapted MPD")
    plt.grid(True, which='both', linestyle='--')
    plt.show()

    return mpd_vals

def plot_mpd_convergence_comparison(num_ls, d=1, rho=5, sigma=1, lbd=-50, ubd=50, gamma=1, seed=42, mpd_vals=None, smoothed_mpd_vals=None):
    """
    Compare convergence of adapted MPD vs smoothed MPD across sample sizes.
    Allows reuse of existing results if provided.
    """
    recompute_mpd = mpd_vals is None or len(mpd_vals) != len(num_ls)
    recompute_smooth = smoothed_mpd_vals is None or len(smoothed_mpd_vals) != len(num_ls)

    if recompute_mpd:
        mpd_vals = []
    if recompute_smooth:
        smoothed_mpd_vals = []

    if recompute_mpd or recompute_smooth:
        for idx, n in enumerate(tqdm(num_ls, desc="Computing MPD")):
            X, Y = generate_uniform_martingale_coupling(n_samples=n, d=d, seed=seed)
            if recompute_smooth:
                params = get_params(rho, X, Y, sigma)
                smoothed_mpd = mtgl_proj_mc(params, lbd, ubd)
                smoothed_mpd_vals.append(smoothed_mpd)
            if recompute_mpd:
                if d ==1:
                    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
                mpd = compute_adapted_mpd(X, Y, gamma=gamma)
                mpd_vals.append(mpd)

    # Linear plot
    plt.figure(figsize=(6, 4))
    if mpd_vals: plt.plot(num_ls, mpd_vals, marker='o', label='Adapted MPD')
    if smoothed_mpd_vals: plt.plot(num_ls, smoothed_mpd_vals, marker='^', label='Smoothed MPD')
    plt.xlabel("Number of Samples")
    plt.ylabel("MPD Value")
    plt.title("Convergence of Adapted MPD vs Smoothed MPD")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Log-log plot
    plt.figure(figsize=(6, 4))
    if mpd_vals: plt.loglog(num_ls, mpd_vals, marker='o', label='Adapted MPD')
    if smoothed_mpd_vals: plt.loglog(num_ls, smoothed_mpd_vals, marker='^', label='Smoothed MPD')

    if mpd_vals:
        ref_x = np.array(num_ls)
        ref_y = ref_x ** (-0.5) * mpd_vals[0] / (num_ls[0] ** -0.5)
        plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/2}$ reference")

    plt.xlabel("log(Number of Samples)")
    plt.ylabel("log(MPD Value)")
    plt.title("Log-Log Convergence Comparison")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()

    return mpd_vals, smoothed_mpd_vals



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
