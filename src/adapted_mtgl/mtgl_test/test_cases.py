import numpy as np
import scipy as scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.adapted_mtgl.mtgl_test.mtgl import mtgl_test, mtgl_test_mc, cutoff
from src.adapted_mtgl.mtgl_test.tools import get_params

# Part (i): test cases for mtgl projection test

def Hermite_y(x,k,plot = False):
    result = x+ (1/np.sqrt(scipy.special.factorial(k)))*scipy.special.eval_hermitenorm(k, x)
    if plot:
        plt.plot(result)
        plt.xlabel("Step")
        plt.ylabel("Position")
        plt.title("Hermite sequence")
    return result

def random_walk(n_steps):
    # Generate random errors following standard normal distribution
    errors = np.random.standard_normal(size=n_steps)  
    # Accumulate errors to create the random walk
    walk = np.cumsum(errors)

    # Plot the random walk
    plt.plot(walk)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Random Walk with Standard Normal Errors")
    plt.show()
    
    return walk

def random_walk_drift(n_steps, mu0, gamma):
    # Generate random errors following standard normal distribution
    errors = np.random.standard_normal(size=n_steps)
    
    # Calculate drift component
    drift_component = mu0*np.float_power(n_steps,-gamma)
    
    # Combine errors and drift to create the random walk
    walk = np.cumsum(errors + drift_component)
    
    plt.plot(walk)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Random Walk with Standard Normal Errors")
    plt.show()
    
    return walk


def simulate_explosive_ar(n_steps, alpha):
    # Generate random noise following standard normal distribution
    noise = np.random.standard_normal(size=n_steps)
    
    # Initialize sequence with a random starting value
    sequence = [np.random.random()]
    
    # Generate the sequence based on the explosive AR process
    for i in range(1, n_steps):
        # Calculate the next value using the AR model
        next_value = alpha * sequence[i-1] + noise[i]
        
        # Append the next value to the sequence
        sequence.append(next_value)
        
    plt.plot(sequence)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Random Walk with Constand Drift and Standard Normal Errors")
    plt.show()
    
    return sequence


def simulate_arma(n_steps, ar_coef, ma_coef):
    # Generate random noise following standard normal distribution
    noise = np.random.standard_normal(size=n_steps)
    
    # Initialize the sequence with zeros
    sequence = np.zeros(n_steps)
    
    # Generate the ARMA sequence
    for t in range(1, n_steps):
        sequence[t] = ar_coef * sequence[t-1] + ma_coef * noise[t-1] + noise[t]
    
    plt.plot(sequence)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("ARMA(1,1) Sequence")
    plt.show()
    
    return sequence


# Part (ii). Plot asymptotic rejection rate and power curves
def asymp_rej_Hermite(k,n_sim,n,rho,lbd,ubd,conf,result, seed = 42): # n is n samples from the coupling series
    ls = []
    mean_ls = []
    for i in tqdm(range(n_sim), desc="Running simulations..."):
        np.random.seed(seed+i)
        x = np.random.normal(0,1,n)
        y = Hermite_y(x,k)
        params = get_params(rho, x, y)
        outcome = mtgl_test(params,lbd,ubd,conf,result, display = False)
        mean_ls.append(outcome[0])
        if not outcome[1]:
            ls.append(1)
    # return asymptotic rejection rate
    count_false = sum(ls)
    return count_false/n_sim, mean_ls

def asymp_rej_Hermite_mc(k,n_sim,n,rho,lbd,ubd,conf,result, seed = 42):
    ls = []
    mean_ls = []
    for i in tqdm(range(n_sim), desc="Running simulations..."):
        np.random.seed(seed+i)
        x = np.random.normal(0,1,n)
        y = Hermite_y(x,k)
        params = get_params(rho, x, y)
        outcome = mtgl_test_mc(params,lbd,ubd,conf,result, display = False) # seed is fixed for mc integration evaluation
        mean_ls.append(outcome[0])
        if not outcome[1]:
            ls.append(1)
    # return asymptotic rejection rate
    count_false = sum(ls)
    return count_false/n_sim, mean_ls


def asymp_rej_basic(p,n_sim,n,rho,lbd,ubd,conf,result,seed = 42): # p is perturbation
    ls = []
    mean_ls = []
    for i in tqdm(range(n_sim), desc="Running simulations..."):
        np.random.seed(seed+i)
        x = np.random.normal(0,1,n)
        z = np.random.normal(0,1,n)
        y = x+z+p
        params = get_params(rho,x,y)
        outcome = mtgl_test(params,lbd,ubd,conf,result, display = False)
        mean_ls.append(outcome[0])
        if not outcome[1]:
            ls.append(1)
    # return asymptotic rejection rate
    count_false = sum(ls)
    return count_false/n_sim, mean_ls


def asymp_rej_basic_mc(p,n_sim,n,rho,lbd,ubd,conf,result,seed = 42):
    ls = []
    mean_ls = []
    for i in tqdm(range(n_sim), desc="Running simulations..."):
        np.random.seed(seed+i)
        x = np.random.normal(0,1,n)
        z = np.random.normal(0,1,n)
        y = x+z+p
        params = get_params(rho,x,y)
        outcome = mtgl_test_mc(params,lbd,ubd,conf,result, display = False)
        mean_ls.append(outcome[0])
        if not outcome[1]:
            ls.append(1)
    # return asymptotic rejection rate
    count_false = sum(ls)
    return count_false/n_sim, mean_ls


def power_curve_basic(perturbation_grid, n_sim,n,rho,lbd,ubd,conf,result,seed = 42):
    rej_ls = []
    mean = []
    for i in range(len(perturbation_grid)):
        rej, mean_ls = asymp_rej_basic(perturbation_grid[i],n_sim,n,rho,lbd,ubd,conf,result,seed)
        rej_ls.append(rej)
        mean.append(np.array(mean_ls).mean())

    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,rej_ls, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Rejection rate")
    plt.title("Power Curve")
    plt.grid(True)

    # plot mean statistics
    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,mean, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Mean Test Statistics")
    plt.title("Average Test Statistics vs. Perturbation")
    plt.grid(True)

    plt.show()


def power_curve_basic_mc(perturbation_grid, n_sim,n,rho,lbd,ubd,conf,result,seed = 42):
    rej_ls = []
    mean = []
    for i in range(len(perturbation_grid)):
        rej, mean_ls = asymp_rej_basic_mc(perturbation_grid[i],n_sim,n,rho,lbd,ubd,conf,result,seed)
        rej_ls.append(rej)
        mean.append(np.array(mean_ls).mean())

    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,rej_ls, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Rejection rate")
    plt.title("Power Curve (MC version)")
    plt.grid(True)

    # plot mean statistics
    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,mean, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Mean Test Statistics")
    plt.title("Average Test Statistics vs. Perturbation (MC version)")
    plt.grid(True)

    plt.show()


def power_curve_Hermite(perturbation_grid, n_sim,n,rho,lbd,ubd,conf,result,seed = 42):
    rej_ls = []
    mean = []
    n_grid = len(perturbation_grid)
    for i in range(n_grid):
        rej, mean_ls = asymp_rej_Hermite(perturbation_grid[i],n_sim,n,rho,lbd,ubd,conf,result,seed)
        rej_ls.append(rej)
        mean.append(np.array(mean_ls).mean())

    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,rej_ls, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Rejection rate")
    plt.title("Power Curve")
    plt.grid(True)

    # plot mean statistics
    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,mean, 'bo-')
    plt.plot(perturbation_grid,[cutoff(result,conf)]*n_grid,'g',label=r"Critical value at $\alpha=0.05$")
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Mean Test Statistics")
    plt.title("Average Test Statistics vs. Perturbation")
    plt.grid(True)
    plt.show()


def power_curve_basic_mc(perturbation_grid, n_sim,n,rho,lbd,ubd,conf,result,seed = 42):
    rej_ls = []
    mean = []
    for i in range(len(perturbation_grid)):
        rej, mean_ls = asymp_rej_Hermite_mc(perturbation_grid[i],n_sim,n,rho,lbd,ubd,conf,result,seed)
        rej_ls.append(rej)
        mean.append(np.array(mean_ls).mean())

    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,rej_ls, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Rejection rate")
    plt.title("Power Curve (MC version)")
    plt.grid(True)

    # plot mean statistics
    plt.figure(figsize=(6, 4))
    plt.plot(perturbation_grid,mean, 'bo-')
    plt.xlabel("Shift / Perturbation magnitude")
    plt.ylabel("Mean Test Statistics")
    plt.title("Average Test Statistics vs. Perturbation (MC version)")
    plt.grid(True)

    plt.show()
