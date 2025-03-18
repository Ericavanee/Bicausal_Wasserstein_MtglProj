"""
This script simulates: 
    (i). smoothed density of a Brownian Motion given different values of sigmas.
    (ii). error of martigale projection given samples size.

Here the martingale coupling is taken from a Brownian Motion, where Y-X is assumed to be independent from X.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal, norm
import warnings

warnings.filterwarnings("ignore")


# Brownian Motion Simulation
def plot_brownian_motion(n, seed = 42):
    np.random.seed(seed)
    x = np.cumsum(np.random.randn(n))
    y = np.cumsum(np.random.randn(n))
    # We add 10 intermediary points between two successive points. We interpolate x and y
    k = 10
    x2 = np.interp(np.arange(n * k), np.arange(n) * k, x)
    y2 = np.interp(np.arange(n * k), np.arange(n) * k, y)
    fig, ax = plt.subplots(1, 1)
    # Now, we draw our points with a gradient of colors.
    ax.scatter(x2, y2, c=range(n * k), linewidths=0,
            marker='o', s=3, cmap=plt.cm.jet,)
    ax.axis('equal')
    ax.set_axis_off()
    ax.set_title('Simulated Brownian Motion')
    plt.show()

# (i). Simulate smoothed density of a Brownian Motion given different values of sigmas
def get_multivariate_normal(mean,cov):
    return multivariate_normal(mean, cov)

def post_smooth_density(xb,yb,x,y,w_rv): # w_rv is a scipy.stats.multivariate_normal object
    n = len(x)
    sum_ls = []
    for i in range(n):
        emp = [xb-x[i],yb-y[i]]
        temp = w_rv.pdf(emp)
        sum_ls.append(temp)
    return (1/n)*sum(sum_ls)


def plot_smooth_density(x,y,p,lbd,ubd,w_rv):
    fig = plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    X = np.linspace(lbd,ubd,p)
    Y = np.linspace(lbd,ubd,p)
    X, Y = np.meshgrid(X, Y)
    Z = []
    for i in range(p):
        row_ls = []
        for j in range(p):
            row_ls.append(post_smooth_density(X[i][j],Y[i][j],x,y,w_rv))
        Z.append(row_ls)
    Z = np.array(Z).reshape(p,p)
        
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
# (ii). Error of martigale projection given samples size
# fomularic integration
def compute_condExp_y(xb,x,y,sig):
    n = len(x)
    normal_rv = norm(0,1)
    # get numerator
    num_sum_ls = []
    for i in range(n):
        temp = (y[i]-x[i]+xb)*normal_rv.pdf((xb-x[i])/sig)
        num_sum_ls.append(temp)
    numerator = sum(num_sum_ls)
    # get denominator
    den_sum_ls = []
    for i in range(n):
        temp = normal_rv.pdf((xb-x[i])/sig)
        den_sum_ls.append(temp)
    denominator = sum(den_sum_ls)
    # get conditional expectation
    condExp = numerator/denominator
    return condExp


def plot_conditionalExp_y(x,y,p,sig):
    x_grid = np.linspace(-5,5,p)
    y_ls = []
    for i in range(p):
        y_ls.append(compute_condExp_y(x_grid[i],x,y,sig))
    
    plt.title("Plot E[Y|X] vs X")
    plt.xlabel("X val")
    plt.ylabel("E[Y|X]")
    plt.plot(x_grid,y_ls)
    plt.plot(x_grid,x_grid)
    plt.show()
    
    return y_ls


# Monte Carlo Simulation

def get_XY_prime(xb,yb,x,y,sig):
    condExp = compute_condExp_y(xb,x,y,sig)
    x_prime = xb+(1/2)*(condExp-xb)
    y_prime = yb+(1/2)*(xb-condExp)
    return [x_prime,y_prime]

def calc_obj(xb,yb,x,y,sig):
    x_prime,y_prime = get_XY_prime(xb,yb,x,y,sig)
    obj = np.linalg.norm(x=[xb-x_prime],ord=2)**2+np.linalg.norm(x=[yb-y_prime],ord=2)**2
    return obj

def monteCarlo_obj(x,y,n,sig, seed = 42):
    np.random.seed(seed)
    # truncate x,y
    x = x[:n] # first n elements
    y = y[:n] # first n elements
    # create smoothed vec
    z1 = np.random.normal(0,1,100*n)
    z2 = np.random.normal(0,1,100*n)
    xb = []
    for i in range(100*n):
        x_temp = np.random.choice(x)
        xb.append(x_temp+sig*z1[i])
    
    yb = []
    for i in range(100*n):
        y_temp = np.random.choice(y)
        yb.append(y_temp+sig*z1[i]+sig*z2[i])
    
    #calculate objective function
    sum_ls = []
    for i in range(100*n):
        temp = calc_obj(xb[i],yb[i],x,y,sig)
        sum_ls.append(temp)
        
    return (1/(100*n))*sum(sum_ls)   


def plot_monteCarlo(sig,n_grid):
    p = len(n_grid)
    
    # create unsmoothed vec
    x = np.random.normal(0,1,n_grid[p-1])
    z = np.random.normal(0,1,n_grid[p-1]) # z = y-x
    y = np.add(z, x)
    
    # create plots
    fig = plt.figure(figsize=(12,10)) # set size
    y_ls = []
    for i in range(p):
        print(n_grid[i])
        temp = monteCarlo_obj(x,y,n_grid[i],sig)
        print(temp)
        y_ls.append(temp)
        print()
    
    plt.plot(n_grid,y_ls, 'ro')
    plt.plot(n_grid,y_ls)
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.title("Monte Carlo Simulation of Error given Sample Size")
    plt.show()
    
    return [n_grid,y_ls]






