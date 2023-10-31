import numpy as np
from scipy import special
import scipy as scipy
import matplotlib.pyplot as plt

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