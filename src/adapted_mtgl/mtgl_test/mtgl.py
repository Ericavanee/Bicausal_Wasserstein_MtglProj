"""
Implements Multi-dimensional Martingale test for gamma = 1.
Includes the simulation of the asymptotic distribution of the martingale test test satistic.
"""

import warnings
from src.adapted_mtgl.mtgl_test.test_cases import *
from src.adapted_mtgl.mtgl_test.multiD import *

warnings.filterwarnings("ignore")



# Part 1: helper functions

def cutoff(result, conf):
    percentile_cutoff = np.percentile(result, conf)
    print(f"cutoff value: {percentile_cutoff}")
    return percentile_cutoff

def get_bounds(domain, d):
    lbd, ubd = domain
    bounds = [(lbd, ubd) for _ in range(d)]
    return bounds

# Part 2: mtgl test

def kernel(rho, sigma, x):
    try:
        d = len(x)
    except:
        d = 1
    return sigma**(-d)*(((rho-1)/2)*(np.float_power(np.linalg.norm(x)+1,-rho)))

def get_params(rho,x,y,sigma):
    params = {
    'rho': rho,
    'x': x,
    'y': y,
    'sigma': sigma}
    return params 

def integrand(vars,params):
    rho = params['rho']
    sigma = params['sigma']
    x = params['x']
    y = params['y']
    n = len(x)
    sum_ls = []
    for i in range(n):
        sum_ls.append(np.multiply(y[i]-x[i],kernel(rho,sigma,x-x[i])))
    integrand = (1/n)*np.linalg.norm(sum(sum_ls))
    return integrand


def mtgl_proj(params,lbd,ubd):
    warnings.simplefilter("ignore")
    x = params['x']
    try:
        d=len(x[0])
    except:
        d = 1
    if d==1:
        f = lambda *vars: integrand(vars,params)
        I = quad(f, lbd, ubd)
    else:
        bounds = get_bounds([lbd,ubd],d)
        f = lambda *vars: integrand(vars,params)
        I = nquad(f, bounds)
    return I[0]


def mtgl_test(params,lbd,ubd,conf,result):
    bond = cutoff(result,conf)
    x = params['x']
    n = len(x)
    proj = np.sqrt(n)*mtgl_proj(params,lbd,ubd)
    if proj <= bond:
        print("Accept null hypothesis with ", conf, "% confidence.")
        return proj, True
    else:
        print("Reject null hypothesis with ", conf, "% confidence.")
        return proj, False
    
def asymp_rej_rate(n_sim,params,lbd,ubd,conf,result):
    ls = []
    for i in range(n_sim):
        if not mtgl_test(params,lbd,ubd,conf,result)[1]:
            ls.append(1)
    # return asymptotic rejection rate
    count_false = sum(ls)
    return count_false/n_sim


def power_curve(grid,n_sim,params,lbd,ubd,conf,result):
    rej_rate = []
    rho = params['rho']
    sigma = params['sigma']
    x = params['x']
    y = params['y']  
    p = len(grid)
    for i in range(p):
        y_copy = y+grid[i]
        params = get_params(rho,x,y_copy,sigma)
        rej_rate.append(asymp_rej_rate(n_sim,params,lbd,ubd,conf,result))
    
    plt.plot(grid,rej_rate)
    plt.show()
    return grid, rej_rate
    