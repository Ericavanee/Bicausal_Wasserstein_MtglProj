"""
This script implements other tools one can experiment with when using the martingale test framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


# (i). Implement the adapted empirical measure from https://arxiv.org/pdf/2002.07261
def generate_samples(N, d, T):
    """Generate N i.i.d. samples X^n = (X^n_1, ..., X^n_T) in [0,1]^d."""
    return np.random.rand(N, T, d)

def compute_partition_grid(N, d, T):
    """Compute partition grid centers for adapted empirical measure."""
    r = (T + 1) ** -1 if d == 1 else (d * T) ** -1
    step_size = N ** -r
    grid_points = np.arange(0, 1 + step_size, step_size)
    grid_centers = (grid_points[:-1] + grid_points[1:]) / 2
    return grid_centers

def map_to_grid(samples, grid_centers):
    """Map samples to their nearest grid center."""
    tree = KDTree(grid_centers.reshape(-1, 1))
    mapped_samples = np.array([grid_centers[tree.query(sample)[1]] for sample in samples.flatten()])
    return mapped_samples.reshape(samples.shape)

def compute_adapted_empirical_measure(mapped_samples):
    """Compute the adapted empirical measure as a probability mass function."""
    unique, counts = np.unique(mapped_samples, return_counts=True)
    probabilities = counts / mapped_samples.size # this should be equal to N
    return unique, probabilities

def plot_measures(samples, mapped_samples, unique, probabilities):
    """Plot original vs adapted empirical measure."""
    plt.figure(figsize=(10, 4))
    
    # Original empirical measure
    plt.subplot(1, 2, 1)
    plt.hist(samples.flatten(), bins=10, alpha=0.7, color='blue', density=True, label='Empirical')
    plt.xlabel("Sample Values")
    plt.ylabel("Density")
    plt.title("Empirical Measure")
    
    # Adapted empirical measure
    plt.subplot(1, 2, 2)
    plt.bar(unique, probabilities, width=0.05, color='red', alpha=0.7, label='Adapted')
    plt.xlabel("Grid Centers")
    plt.ylabel("Probability")
    plt.title("Adapted Empirical Measure")
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    d, T, N = 1, 2, 8
    samples = generate_samples(N, d, T)
    grid_centers = compute_partition_grid(N, d, T)
    mapped_samples = map_to_grid(samples, grid_centers)
    unique, probabilities = compute_adapted_empirical_measure(mapped_samples)
    plot_measures(samples, mapped_samples, unique, probabilities)