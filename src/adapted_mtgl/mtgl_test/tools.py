"""
This script implements other tools one can experiment with when using the martingale test framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# utils for general
def get_params(rho,x,y,sigma = 1):
    params = {
    'rho': rho,
    'x': x,
    'y': y,
    'sigma': sigma}
    return params 


# (i). Implement the adapted empirical measure from https://arxiv.org/pdf/2002.07261
# do the convergence plot of the MPDs
# uniform on interval (-1/2,1/2) = X, Z; Y = X+Z, X,Z iid
def generate_samples(N, d, T, seed = 42):
    """Generate N i.i.d. samples X^n = (X^n_1, ..., X^n_T) in [0,1]^d."""
    np.random.seed(seed)
    return np.random.rand(N, T, d)

def compute_partition_grid(N, d, T):
    """Compute partition grid centers for adapted empirical measure."""
    r = (T + 1) ** -1 if d == 1 else (d * T) ** -1
    step_size = N ** -r
    grid_1d = np.arange(0, 1 + step_size, step_size)
    grid_centers_1d = (grid_1d[:-1] + grid_1d[1:]) / 2
    if d == 1:
        return grid_centers_1d
    else:
        # Create a Cartesian product grid in d-dimensions
        grids = np.meshgrid(*([grid_centers_1d] * d), indexing='ij')
        grid_centers = np.stack(grids, axis=-1).reshape(-1, d)
        return grid_centers

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

if __name__ == '__main__':
    d, T, N = 2, 2, 27
    samples = generate_samples(N, d, T)
    grid_centers = compute_partition_grid(N, d, T)
    print(grid_centers)
    mapped_samples = map_to_grid(samples, grid_centers)
    unique, probabilities = compute_adapted_empirical_measure(mapped_samples)
    plot_measures(samples, mapped_samples, unique, probabilities)