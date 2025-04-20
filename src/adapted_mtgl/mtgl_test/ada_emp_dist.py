"""
This script implements the adapted empirical measure from https://arxiv.org/pdf/2002.07261
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree # finds nearest neighbor (centers) to group samples
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from src.adapted_mtgl.utils import get_params
from src.adapted_mtgl.mtgl_test.mtgl_couplings import generate_uniform_martingale_coupling
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



def empirical_k_means_measure(data, use_klist=0, klist=(), tol_decimals=6, use_weights=0, heuristic=0): # adapted directly from 
    # use kmenas clustering to come up with center rather than using grid
    (k, T_h) = data.shape
    if not use_klist:
        klist = (np.ones(T_h) * int(np.round(np.sqrt(k)))).astype(int)

    label_list = []
    support_list = []
    out_x = np.zeros([0, T_h])
    out_w = []

    if heuristic:
        for t in range(T_h):
            data_t = data[:, t]
            inds_sort_t = np.argsort(data_t)
            datas_t = data_t[inds_sort_t]
            n_av = int(np.round(k / klist[t]))
            lmax = int(np.floor(n_av * klist[t]))
            all_but_end = np.reshape(datas_t[:lmax], (-1, n_av))
            mean_all_but = np.mean(all_but_end, axis=1, keepdims=1)
            cx = mean_all_but
            mean_all_but = np.tile(mean_all_but, (1, n_av))
            mean_all_but = np.reshape(mean_all_but, (-1, 1))
            mean_rest = np.mean(datas_t[lmax:])
            if lmax < k:
                mean_vec = np.concatenate([np.squeeze(mean_all_but), np.array([mean_rest])])
                cx = np.concatenate([cx, np.array([mean_rest])])
            else:
                mean_vec = np.squeeze(mean_all_but)
            lx = np.zeros(k, dtype=int)
            for i in range(k):
                for j in range(len(cx)):
                    if mean_vec[inds_sort_t[i]] == cx[j]:
                        lx[i] = j
                        continue
            label_list.append(lx)
            support_list.append(cx)

    else:
        for t in range(T_h):
            data_t = data[:, t:t+1]
            kmx = KMeans(n_clusters=klist[t]).fit(data_t)
            cx = kmx.cluster_centers_
            cx = np.round(cx, decimals=tol_decimals)
            lx = kmx.labels_
            label_list.append(lx)
            support_list.append(cx)

    if use_weights == 0:
        out = np.zeros([k, T_h])
        for t in range(T_h):
            out[:, t] = support_list[t][label_list[t]][:, 0]
        return out

    for i in range(k):
        cur_path = np.zeros(T_h)
        for t in range(T_h):
            cur_path[t] = support_list[t][label_list[t][i]]

        path_is_here = 0
        for j in range(len(out_w)):
            if np.all(out_x[j, :] == cur_path):
                out_w[j] += 1 / k
                path_is_here = 1
                break
        if not path_is_here:
            out_x = np.append(out_x, np.expand_dims(cur_path, axis=0), axis=0)
            out_w.append(1 / k)

    return out_x, out_w


def compute_adapted_mpd(X, Y, method = 'grid', gamma=1):
    """
    Compute MPD(P, gamma) = 2^{1 - gamma} E_{\hat{\mu}}[ ||\varphi^N(X) - E_{\hat{\mu}}[\varphi^N(Y) | \varphi^N(X)]||^gamma ]
    where the expectation is taken under the adapted empirical measure based on mapped X and mapped Y.
    This version supports multi-dimensional X and Y.
    Uses K-means adapted empirical measure for flexible partitioning.
    """
    n, d = X.shape

    if method == 'kmeans':
        X_Y = np.concatenate([X, Y], axis=1)  # shape (n, 2d)
        mapped_XY = empirical_k_means_measure(X_Y, use_weights=0)  # shape (n, 2d)

        mapped_X = mapped_XY[:, :d]  # shape (n, d)
        mapped_Y = mapped_XY[:, d:]  # shape (n, d)
    elif method == 'grid':
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
    else:
        raise Exception("Choose either 'grid' or 'kmenas' for method.")

    grouped = defaultdict(list)
    for i in range(n):
        key = tuple(mapped_X[i])
        grouped[key].append(mapped_Y[i])

    mpd_sum = 0.0
    for key, y_vals in grouped.items():
        g = np.array(key)
        y_mean = np.mean(y_vals, axis=0)
        dist = np.linalg.norm(g - y_mean) ** gamma
        mpd_sum += len(y_vals) * dist

    return 2 ** (1 - gamma) * mpd_sum / n


def plot_mpd_convergence(num_ls, type='adapted', method='grid', d=1, rho = 5, sigma = 1, gamma=1, lbd = -50, ubd = 50, seed=0, mpd_vals=None, n_trials=1, regress=False):
    mpd_vals = np.array(mpd_vals).tolist()
    if not mpd_vals:
        mpd_vals = []
        if type == 'adapted':
            for n in tqdm(num_ls, desc="Computing MPD"):
                trial_mpd_vals = []
                for t in range(n_trials):
                    trial_seed = seed + t  # ensure different seeds for each trial
                    X, Y = generate_uniform_martingale_coupling(n_samples=n, d=d, seed=trial_seed)
                    if d == 1:
                        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
                    mpd = compute_adapted_mpd(X, Y, method=method, gamma=gamma)
                    trial_mpd_vals.append(mpd)
                avg_mpd = np.mean(trial_mpd_vals)
                mpd_vals.append(avg_mpd)
        elif type == 'smoothed':
            for n in tqdm(num_ls, desc="Computing Smoothed MPD"):
                trial_mpd_vals = []
                for t in range(n_trials):
                    trial_seed = seed + t
                    X, Y = generate_uniform_martingale_coupling(n_samples=n, d=d, seed=trial_seed)
                    params = get_params(rho, X, Y, sigma)
                    smoothed_mpd = mtgl_proj_mc(params, lbd, ubd, disable_tqdm=True)
                    trial_mpd_vals.append(smoothed_mpd)
                avg_mpd = np.mean(trial_mpd_vals)
                mpd_vals.append(avg_mpd)
        else:
            raise ValueError("Invalid type. Choose either 'adapted' or 'smoothed'.")

    # --- Linear plot ---
    plt.figure(figsize=(6, 4))
    plt.plot(num_ls, mpd_vals, marker='o')
    plt.xlabel("Number of Samples")
    plt.ylabel("MPD Value")
    if type == 'adapted':
        plt.title("Convergence of Adapted MPD vs Sample Size")
    elif type == 'smoothed':
        plt.title("Convergence of Smoothed MPD vs Sample Size")
    plt.grid(True)
    plt.show()

    # --- Log-log plot ---
    plt.figure(figsize=(6, 4))
    if type == 'adapted':
        plt.loglog(num_ls, mpd_vals, marker='o', label='Adapted Empirical MPD')
    elif type == 'smoothed':
        plt.loglog(num_ls, mpd_vals, marker='^', label='Smoothed Empirical MPD')
    lines = []

    if regress:
        # Fit regression in log-log space
        log_n = np.log(num_ls).reshape(-1, 1)
        log_mpd = np.log(mpd_vals).reshape(-1, 1)
        reg = LinearRegression().fit(log_n, log_mpd)
        slope = reg.coef_[0, 0]
        fitted_log_mpd = reg.predict(log_n)
        fitted_mpd = np.exp(fitted_log_mpd.flatten())

        # Plot fitted regression line
        plt.loglog(num_ls, fitted_mpd, linestyle='-', color='pink',
                   label=fr"Fitted Regression Line")
        print(f"Estimated empirical convergence rate (slope): {slope:.4f}")

    # Add theoretical reference lines
    ref_x = np.array(num_ls)
    ref_y = ref_x ** (-0.5) * mpd_vals[0] / (num_ls[0] ** -0.5)
    ref_line1, = plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/2}$ reference")
    lines.append(ref_line1)

    if type == 'adapted':
        if d == 1:
            rate = -1 / 3
            ref_y = ref_x ** rate * mpd_vals[0] / (num_ls[0] ** rate)
            ref_line2, = plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/(T+1)}$ reference")
            lines.append(ref_line2)
        elif d == 2:
            rate = -1 / 4
            ref_y = (ref_x ** rate) * np.log(ref_x + 1)
            ref_y *= mpd_vals[0] / ((num_ls[0] ** rate) * np.log(num_ls[0] + 1))
            ref_line2, = plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/4} \log(n+1)$ reference")
            lines.append(ref_line2)
        elif d >= 3:
            rate = -1 / (2 * d)
            ref_y = ref_x ** rate * mpd_vals[0] / (num_ls[0] ** rate)
            ref_line2, = plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/(2d)}$ reference")
            lines.append(ref_line2)

    plt.xlabel("log(Number of Samples)")
    plt.ylabel("log(MPD Value)")
    if type == 'adapted':
        plt.title("Log-Log Convergence of Adapted MPD" + (" with Regression" if regress else ""))
    elif type == 'smoothed':
        plt.title("Log-Log Convergence of Smoothed MPD" + (" with Regression" if regress else ""))
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc='best')
    plt.show()

    return mpd_vals

def plot_mpd_convergence_comparison(num_ls,
                                    adapted_mpd_vals, smoothed_mpd_vals, d, regress = False):
    """
    Compare convergence of adapted MPD vs smoothed MPD across sample sizes using existing results.
    """
    # Linear plot
    plt.figure(figsize=(6, 4))
    # if mpd_vals:
    plt.plot(num_ls, adapted_mpd_vals, marker='o', label='Adapted MPD')
    # if smoothed_mpd_vals:
    plt.plot(num_ls, smoothed_mpd_vals, marker='^', label='Smoothed MPD')
    plt.xlabel("Number of Samples")
    plt.ylabel("MPD Value")
    plt.title("Convergence of Adapted MPD vs Smoothed MPD")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Log-log plot
    plt.figure(figsize=(6, 4))
    
    plt.loglog(num_ls, adapted_mpd_vals, marker='o', label='Adapted MPD')
    plt.loglog(num_ls, smoothed_mpd_vals, marker='^', label='Smoothed MPD')

    if regress:
        # Fit regression in log-log space
        log_n = np.log(num_ls).reshape(-1, 1)
        log_mpd = np.log(adapted_mpd_vals).reshape(-1, 1)
        reg = LinearRegression().fit(log_n, log_mpd)
        slope = reg.coef_[0, 0]
        fitted_log_mpd = reg.predict(log_n)
        fitted_mpd = np.exp(fitted_log_mpd.flatten())

        # Plot fitted regression line
        plt.loglog(num_ls, fitted_mpd, linestyle='-', color='pink',
                   label=fr"Fitted Regression Line (Adapted)")
        print(f"Estimated adapted MPD empirical convergence rate (slope): {slope:.4f}")
        
        log_mpd = np.log(smoothed_mpd_vals).reshape(-1, 1)
        reg = LinearRegression().fit(log_n, log_mpd)
        slope = reg.coef_[0, 0]
        fitted_log_mpd = reg.predict(log_n)
        fitted_mpd = np.exp(fitted_log_mpd.flatten())

        # Plot fitted regression line
        plt.loglog(num_ls, fitted_mpd, linestyle='-', color='red',
                   label=fr"Fitted Regression Line (Smoothed)")
        print(f"Estimated smoothed MPD empirical convergence rate (slope): {slope:.4f}")

    # Add theoretical reference lines only for "not regress".
    if not regress:  
        ref_x = np.array(num_ls)
        ref_y = ref_x ** (-0.5) * adapted_mpd_vals[0] / (num_ls[0] ** -0.5)
        plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/2}$ reference")

        if d == 1:
            rate = -1 / 3
            ref_y = ref_x ** rate * adapted_mpd_vals[0] / (num_ls[0] ** rate)
            plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/(T+1)}$ reference")
        elif d == 2:
            rate = -1 / 4
            ref_y = (ref_x ** rate) * np.log(ref_x + 1)
            ref_y *= adapted_mpd_vals[0] / ((num_ls[0] ** rate) * np.log(num_ls[0] + 1))
            plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/4} \log(n+1)$ reference")
        elif d >= 3:
            rate = -1 / (2 * d)
            ref_y = ref_x ** rate * adapted_mpd_vals[0] / (num_ls[0] ** rate)
            plt.loglog(ref_x, ref_y, linestyle='--', label=r"$n^{-1/(2d)}$ reference")

    plt.xlabel("log(Number of Samples)")
    plt.ylabel("log(MPD Value)")
    plt.title("Log-Log Convergence Comparison")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()

    return adapted_mpd_vals, smoothed_mpd_vals




# if __name__ == '__main__':
#     d, T, N = 2, 2, 27
#     samples = generate_samples(N, d, T)
#     grid_centers = compute_partition_grid(N, d, T)
#     print(grid_centers)
#     # mapped_samples = map_to_grid(samples, grid_centers)
#     # unique, probabilities = compute_adapted_empirical_measure(mapped_samples)
#     # plot_measures(samples, mapped_samples, unique, probabilities)
#     #num_ls = np.logspace(start=1, stop=6, num=6, base=10, dtype=int)
#     num_ls = np.arange(10,1000+1,100)
#     print(num_ls)
#     plot_adapted_mpd_convergence(num_ls)
