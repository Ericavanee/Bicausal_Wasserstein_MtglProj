from src.adapted_mtgl.mtgl_test.mtgl import *
from src.adapted_mtgl.mtgl_test.multiD import *
from applications.markov.bdd_fun import *

# This package provides functions that implement Section 6.2 of the paper.

# 1.0 Simple Markov chain example

def simple_Markov(T,rho):
    X0 = 0
    ls = [X0]
    for i in range(T):
        error = np.random.randn()
        ls.append(ls[i]*rho+error)
    return ls
    
def cond_expectation_simple(rho, sample_ls, u_list, n_sim = 10000):
    # List to store the resulting d-dimensional vectors
    results = []

    for i in range(len(sample_ls)):
        # Vector to store the conditional expectations for each function u_i
        vector1 = []

        for u in u_list:
            samples = [u(sample_ls[i]*rho+np.random.randn()) for _ in range(n_sim)]
            vector1.append(np.mean(samples))
        
        vector1 = np.array(vector1)

        results.append(vector1)
        
    if len(u_list) == 1:
        return np.array(results).ravel()

    return np.array(results)


def markov_coupling(T, rho, u_list):
    samples = simple_Markov(T+1,rho) # generate W0,...,W_{T+1}
    X = []
    Y_prime = []
    cond_exp = []

    for i in range(T+1):
        # Vector to store the conditional expectations for each function u_i
        vec1 = []
        vec2 = []

        for u in u_list:
            vec1.append(samples[i])
            vec2.append(u(samples[i+1]))

        X.append(vec1)
        Y_prime.append(vec2)
        
    if len(u_list) == 1:
        X = np.array(X).ravel()
        Y_prime = np.array(Y_prime).ravel()
    else:
        X = np.array(X)
        Y_prime = np.array(Y_prime)
    
    Y = X+Y_prime-cond_expectation_simple(rho, samples[:-1], u_list)
    return X,Y


# 2.0 Adapted perpetual cash flow process example

def compound_poisson_process_integrand(t, base_lambda=2, gamma_shape=2, gamma_scale=3):
    # Adjust lambda based on t
    effective_lambda = base_lambda * t
    
    # Generate Poisson distributed number based on the effective lambda
    n_pt = np.random.poisson(effective_lambda)

    # Generate n_pt gamma distributed numbers (jumps)
    jumps = np.random.gamma(gamma_shape, gamma_scale, n_pt)

    # Generate jump times uniformly in [0, t]
    jump_times = np.sort(np.random.uniform(0, t, n_pt))

    return jump_times, jumps


def Z_cont(t, r=1, base_lambda=2, gamma_shape=2, gamma_scale=3):
    jump_times, jumps = compound_poisson_process_integrand(t, base_lambda, gamma_shape, gamma_scale)
    
    # Calculate the contributions for each jump
    contributions = np.exp(r * (jump_times-t)) * jumps

    # Sum the contributions
    Z_t = np.sum(contributions)

    return Z_t


def Z_increment(t, base_lambda=2, gamma_shape=2, gamma_scale=3):
    # Number of jumps in the interval [0, t]
    num_jumps = np.random.poisson(base_lambda * t)

    # Sizes of the jumps
    jump_sizes = np.random.gamma(gamma_shape, gamma_scale, num_jumps)
    
    # Generate jump times uniformly in [0, t]
    jump_times = np.sort(np.random.uniform(0, t, num_jumps))
    
    return np.sum(np.exp(jump_times)*jump_sizes)


def conditional_expectation_for_sample(Zt_samples, u_list, num_simulations=10000):
    # List to store the resulting d-dimensional vectors
    results = []

    for i in range(len(Zt_samples)):
        # Vector to store the conditional expectations for each function u_i
        vector = []

        for u in u_list:
            samples = [u(Zt_samples[i] + Z_increment(1)) for _ in range(num_simulations)]
            vector.append(np.mean(samples))

        results.append(vector)

    return results


# compute the Markov chain given terminal time T
def markov_chain_Z(T):
    Z1 = Z_cont(1)
    chain_ls = [0,Z1]
    for i in range(T-1):
        chain_ls.append(chain_ls[i+1]+Z_increment(1))
    return chain_ls



def transformed_markov_coupling(T, u_list):
    samples = markov_chain_Z(T) # generate W0,...,W_{T+1}
    X = []
    Y_prime = []

    for i in range(T):
        # Vector to store the conditional expectations for each function u_i
        vec1 = []
        vec2 = []

        for u in u_list:
            vec1.append(u(samples[i]))
            vec2.append(u(samples[i+1]))

        X.append(vec1)
        Y_prime.append(vec2)
    
    X = np.array(X)
    Y_prime = np.array(Y_prime)
    
    Y = X+Y_prime-np.array(conditional_expectation_for_sample(samples[:-1], u_list))
    return X,Y



