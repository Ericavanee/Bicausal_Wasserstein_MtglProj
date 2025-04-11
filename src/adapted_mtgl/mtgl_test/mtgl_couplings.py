"""
Generate martigale coupling examples.
"""
import os
import numpy as np

# for discrete example of martingale coupling
def generate_uniform_martingale_coupling(n_samples, d=1, seed=42, save = False, save_dir = "coupling_data"):
    """
    Generate (X, Y) where X ~ U(-1/2, 1/2)^d and Y = X + Z with Z ~ U(-1/2, 1/2)^d
    X, Z independent, so E[Y | X] = X (martingale coupling).
    """
    np.random.seed(seed)
    X = np.random.uniform(-0.5, 0.5, size=(n_samples, d)) # -0.5,0.5
    Z = np.random.uniform(-0.5, 0.5, size=(n_samples, d))
    Y = X + Z
    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"uniform{n_samples}d{d}seed{seed}"
        filepath = os.path.join(save_dir, filename)
        np.savez(filepath, X=X, Y=Y)
    return X, Y

def basic(n_samples=100, seed=42, save=False, save_dir="coupling_data"):
    """
    1D martingale coupling: X ~ Uniform(-1/2, 1/2), Y = X + Z, with Z ~ Uniform(-1/2, 1/2)
    """
    np.random.seed(seed)
    X = np.random.uniform(-0.5, 0.5, size=(n_samples, 1))
    Z = np.random.uniform(-0.5, 0.5, size=(n_samples, 1))
    Y = X + Z

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"basic{n_samples}d1seed{seed}.npz"
        filepath = os.path.join(save_dir, filename)
        np.savez(filepath, X=X, Y=Y)

    return X, Y

# basic helper multi-dimensional martingale coupling generator (X,Y)
def basic_multi(n_samples=100, d=2, seed=42, save=False, save_dir="coupling_data"):
    """
    Multidimensional martingale coupling: X ~ U(-1/2, 1/2)^d, Y = X + Z with Z ~ U(-1/2, 1/2)^d
    """
    np.random.seed(seed)
    X = np.random.uniform(-0.5, 0.5, size=(n_samples, d))
    Z = np.random.uniform(-0.5, 0.5, size=(n_samples, d))
    Y = X + Z

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"basic{n_samples}d{d}seed{seed}.npz"
        filepath = os.path.join(save_dir, filename)
        np.savez(filepath, X=X, Y=Y)

    return X, Y
