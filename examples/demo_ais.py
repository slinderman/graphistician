# Run as script using 'python -m test.synth'
import sys

import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import brewer2mpl

from graph_models import *
from utils import *

def estimate_marginal_likelihood(A, f0, theta0, model,
                                 N_samples=1000, B=100, steps_per_B=11):
    """
    Use AIS to approximate the marginal likelihood of a latent network model
    """
    N = A.shape[0]
    betas = np.linspace(0,1,B)

    # Sample m points
    log_weights = np.zeros(N_samples)
    for m in range(N_samples):
        # Sample a new set of graph parameters from the prior
        f,theta = copy.deepcopy((f0, theta0))

        # print "M: %d" % m
        # Sample mus from each of the intermediate distributions,
        # starting with a draw from the prior.
        samples = []

        # Ratios correspond to the 'f_{n-1}(x_{n-1})/f_{n}(x_{n-1})' values in Neal's paper
        ratios = np.zeros(B-1)

        # Sample the intermediate distributions
        for (b,beta) in zip(range(1,B), betas[1:]):
            # print "M: %d\tBeta: %.3f" % (m,beta)
            sys.stdout.write("M: %d\tBeta: %.3f \r" % (m,beta))
            sys.stdout.flush()

            # Take some number of steps per beta
            for s in range(steps_per_B):
                # Sample the model parameters theta
                theta = model.sample_theta((A,f), beta=beta)

                # Sample features f
                for n in np.arange(N):
                    f[n] = model.sample_f(theta, (n,A,f), beta=beta)


            # Compute the ratio of this sample under this distribution and the previous distribution
            curr_lkhd = model.logpr(A, f, theta, beta=beta)
            prev_lkhd = model.logpr(A, f, theta, beta=betas[b-1])

            ratios[b-1] = curr_lkhd - prev_lkhd

        # Compute the log weight of this sample
        log_weights[m] = np.sum(ratios)

        print ""
        print "W: %f" % log_weights[m]

    # Compute the mean of the weights to get an estimate of the normalization constant
    log_Z = -np.log(N_samples) + logsumexp(log_weights)
    return log_Z


#
# DEMO: Compute marginal likelihood of Erdos Renyi network with AIS
#

# Create two models
N = 16
# Erdos Renyi
a0 = 1.0
b0 = 1.0
er_model = ErdosRenyiNetwork(x=(a0,b0))
# SBM
R = 3
b0 = 1.0
b1 = 1.0
alpha0 = 10*np.ones(R)
sbm_model = StochasticBlockModel(R=R, b0=b0, b1=b1, alpha0=alpha0)

# Sample from the "true" Erdos Renyi model
# (A_true, f_true, theta_true) = sample_network(er_model, N)

# Or sample from a true SBM
(A_true, f_true, theta_true) = sample_network(sbm_model, N)
order = invariant_sbm_order(f_true, R)
A_true = A_true[np.ix_(order, order)]

# Plot the true network
plt.figure()
plt.spy(A_true)
plt.title("True Network")
plt.show(block=False)
raw_input("Press any key to begin AIS...\n")

unif_marg_lkhd = N**2 * np.log(0.5)

# Compute marginal likelihood of an Erdos Renyi model with the same empirical sparsity
nnz_A = float(A_true.sum())
N_conns = A_true.size
eb_rho = nnz_A / N_conns
eb_er_marg_lkhd = nnz_A * np.log(eb_rho) + (N_conns-nnz_A)*np.log(1-eb_rho)
print "Empirical Bayes ER Marg Lkhd: ", eb_er_marg_lkhd

# Estimate the marginal likelihood of ER model with a uniform prior on the sparsity
print "Computing ER Marginal Likelihood"
_, f0, theta0 = sample_network(er_model, N)
est_er_marg_lkhd = estimate_marginal_likelihood(A_true, f0, theta0, er_model,
                                                N_samples=5, steps_per_B=2)
print "Estimated Bayesian ER Marg Lkhd: ", est_er_marg_lkhd

# Estimate the marginal likelihood of a stochastic block model with AIS
print "Computing SBM Marginal Likelihood"
_, f0, theta0 = sample_network(sbm_model, N)
est_sbm_marg_lkhd = estimate_marginal_likelihood(A_true, f0, theta0, sbm_model,
                                                N_samples=5, steps_per_B=2)
print "Estimated SBM Marg Lkhd: ", est_sbm_marg_lkhd

# Plot the marginal likelihood results
plt.figure()
res = np.array([eb_er_marg_lkhd, est_er_marg_lkhd, est_sbm_marg_lkhd]) - unif_marg_lkhd
plt.bar(np.arange(3), res, width=0.8)
plt.xticks(np.arange(3)+0.4)
plt.gca().set_xticklabels(['EB ER', 'Bayes ER', 'Bayes SBM'], rotation='vertical')
plt.title("True Network")
plt.show(block=True)
