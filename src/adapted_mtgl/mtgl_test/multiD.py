"""
This script contains functions for simulating and integrating multi-dimensional Gaussian random fields.
"""
import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.integrate import nquad
from scipy.integrate import quad
import scipy.integrate as integrate


# basic helper 1-dimensional martingale coupling generator (X,Y)
def basic(n_samples, seed = 42):
    np.random.seed(seed)
    x = np.random.normal(0,1,n_samples)
    z = np.random.normal(0,1,n_samples)
    y = np.add(z, x)
    return x,y

# basic helper multi-dimensional martingale coupling generator (X,Y)
def basic_multi(n_samples,d, seed = 42):
    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.eye(d)
    x = np.random.multivariate_normal(mean, cov, n_samples)
    z = np.random.multivariate_normal(mean, cov, n_samples)
    y = np.add(z, x)
    return x,y

def generate_grid_points(dim, n_points, lower_bound, upper_bound):
    """
    Generate grid points in R^d space.
    
    Parameters:
    - dim (int): The dimension of the space.
    - n_points (int): The number of points per dimension.
    - lower_bound (float): The lower bound for all dimensions.
    - upper_bound (float): The upper bound for all dimensions.
    
    Returns:
    - List[Tuple[float, ...]]: A list of tuples representing the grid points.
    """
    # Generate a linear space for one dimension
    one_dim_space = [
        lower_bound + (upper_bound - lower_bound) * i / (n_points - 1) 
        for i in range(n_points)
    ]
    
    # Create grid points using Cartesian product
    grid_points = list(product(one_dim_space, repeat=dim))
    
    if dim == 1:
        grid_points = np.array(grid_points).ravel()
    
    return grid_points


# aka: kernel
# input x is an array
def smoothing_function(rho, sigma, diff):
    try:
        d = len(diff[0])
    except:
        d = 1
    if d==1:
        normed_ls = np.abs(diff)
    else:
        normed_ls = np.array([np.linalg.norm(arr) for arr in diff])
    return sigma**(-d)*(((rho-1)/2)*(np.float_power(normed_ls+1,-rho)))

def smooth1D(sigma, rho, diff):
    n = len(diff)
    d = 1
    
    result = []
    for i in range(n):
        if abs(diff[i]) <= sigma:
            result.append(sigma ** (-d))
        else:
            result.append((sigma ** (rho - d)) * ((rho - 1) / 2)*np.float_power((abs(diff[i]) + 1), -rho))
            
    return result

# for each grid point x \in R^d and y \in R^d, the covariance of GxGy is a d-by-d matrix 
def covariance_fn_multi(x, y, X, Y, sigma, rho):
    n = len(X) # read # of samples
    d = len(X[0])
    # convert to numpy array
    X = np.array(X)
    Y = np.array(Y)
    smooth1 = np.array(smoothing_function(rho,sigma,x - X)).reshape(n,1)
    smooth2 = np.array(smoothing_function(rho,sigma,y - X)).reshape(n,1)
    diff = Y-X # n-by-d matrix
    mat1 = np.array(diff*smooth1).transpose() # d-by-n matrix
    mat2 = diff*smooth2 #n-by-d matrix
    result = np.zeros((d, d, n)) # create tensor
    # compute the list of n d-by-d matrices
    for i in range(n):
        result[:, :, i] = np.outer(mat1[:, i], mat2[i, :])
    #return mean of the n d-by-d matrices
    return np.mean(result, axis=2)

# 1st dimension version
def covariance_fn(x,y,X,Y,sigma,rho):
    n = len(X)
    smooth1 = smooth1D(sigma,rho,x-X)
    smooth2 = smooth1D(sigma,rho,y-X)
    diff = Y-X
    return (1/n)*sum(diff*smooth1*smooth2*diff)


# expand matrix
def expand_matrix(mat):
    return np.block([[mat[i,j] for j in range(mat.shape[1])] for i in range(mat.shape[0])])

# returns a (d*n)-by(d*n) matrix 
def covariance_mat_multi(grid, X, Y, sigma, rho):
    n = len(grid)
    d = len(X[0])
    mat = np.empty((n, n, d, d))
    for i in range(n):
        for j in range(n):
            # Ensure each entry in the covariance matrix is a scalar
            mat[i, j] = covariance_fn_multi(grid[i], grid[j], X, Y, sigma, rho)
    expanded_mat = expand_matrix(mat)
    return expanded_mat


# first dimension version
def covariance_mat(grid,X,Y,sigma,rho):
    n = len(grid)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mat[i,j] = covariance_fn(grid[i],grid[j],X,Y,sigma,rho)     
    return mat


def generate_multi_GRF(grid, X, Y, sigma, rho, n_sim):
    n = len(grid) # number of grid points
    try:
        d = len(X[0])
    except:
        d = 1
    mean = np.zeros((n*d))
    if d == 1:
        mean = np.zeros(n)
        cov_matrix = covariance_mat(grid,X,Y,sigma,rho)
        samples = np.abs(np.random.multivariate_normal(mean, cov_matrix, size = n_sim)) # take the absolute value
        return samples
    else:
        mean = np.zeros((n*d))
        cov_matrix = covariance_mat_multi(grid,X,Y,sigma,rho)
        raw_samples = np.random.multivariate_normal(mean, cov_matrix, size = n_sim)
        # take the L-2 norm
        reshaped_array = raw_samples.reshape(n_sim, n, d)
        samples = np.linalg.norm(reshaped_array, axis=2) # each sample represents the scalar value associated with each grid point
        return samples


# numerical integration methods
# Step 1: Fit RBF
def fit_rbf(data_points, grid_points):
    """
    Fit an RBF function given data points and their corresponding grid points.

    Parameters:
    - data_points: A 1D array-like of scalar data points.
    - grid_points: A 2D array-like where each row is a d-dimensional coordinate.

    Returns:
    - A function representing the RBF interpolation of the data.
    """
    # Use * to unpack the grid points into separate components
    rbf_func = Rbf(*grid_points.T, data_points)
    return rbf_func

# Step 2: Define a recursive function for d-dimensional integration.
# Since nquad takes care of multiple integrations, no recursion is explicitly needed.

# Step 3: Integrate the RBF function over the entire d-dimensional domain.
def integrate_rbf(rbf_func, bounds):
    """
    Integrate the RBF function over a d-dimensional domain.

    Parameters:
    - rbf_func: The RBF function to integrate.
    - bounds: A list of tuple bounds for each dimension.

    Returns:
    - The integrated value over the domain.
    """
    return nquad(rbf_func, bounds)[0]

# simulate 1D asymptotic distribution using scipy.integrate
def simulate_integral_distribution(n, n_sim, domain, sigma, rho, save = True, save_dir = "sim_data", seed = 42):
    # generate base X, Y
    X, Y = basic(n_sim, seed)
    # n: partition size
    xmin, xmax = domain
    grid = np.linspace(xmin,xmax,n+1)
    mean = np.zeros(n+1)
    Z = np.random.normal(size=n+1)
    cov_matrix = covariance_mat(grid,X,Y,sigma,rho)
    samples = np.abs(np.random.multivariate_normal(mean, cov_matrix, size = n_sim)) # take the absolute value
    integral_ls = []
    for i in range(n_sim):
        integral_ls.append(integrate.simps(samples[i], grid))
        
    integral_ls = np.array(integral_ls) # cast as array
    mean_integral = integral_ls.mean()
    var_integral = integral_ls.var()

    print(f"Mean of the integral: {mean_integral}")
    print(f"Variance of the integral: {var_integral}")

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"d1rho{rho}sig{sigma}p{n}n{n_sim}seed{seed}.txt"
        save_path = os.path.join(save_dir, save_name)
        np.savetxt(save_path, integral_ls)

    return integral_ls

# integrate a list of scalar values over a corresponding list of fixed grid points
# includes option for 1D, and provides a variety of methods for integration
def simulate_integral_multi(domain, n_points, X, Y, sigma, rho, n_sim, method = 'trapez'):
    try:
        d = len(X[0])
    except:
        d = 1
    domain_bounds = [domain] * d # common domain
    grid = np.array(generate_grid_points(d, n_points, domain[0], domain[1]))
    samples = generate_multi_GRF(grid, X, Y, sigma, rho, n_sim)
    integral_ls = []
    if d == 1:
        for i in range(n_sim):
            # use quad (highest complexity)
            if method == 'quad':
                func = Rbf(grid, samples[i], function='linear') 
                result, _ = quad(func, domain[0], domain[1])
            # use trap
            elif method == 'trapez':
                result = np.trapz(samples[i],grid)
            elif method == 'simps':
                result = integrate.simps(samples[i], grid)
            else:
                raise ValueError("Method must be either 'quad', 'trapez,' or 'simps.'")
            integral_ls.append(result)
    else:        
        for i in range(n_sim):
            # Interpolate using RBF
            func = fit_rbf(samples[i], grid)
            result = integrate_rbf(func, domain_bounds)
            integral_ls.append(result)
        
    integral_ls = np.array(integral_ls) # cast as array
    mean_integral = np.mean(integral_ls)
    var_integral = np.var(integral_ls)
    
    print(f"Mean of the integral: {mean_integral}")
    print(f"Variance of the integral: {var_integral}") 
    return integral_ls



