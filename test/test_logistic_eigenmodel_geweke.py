"""
Geweke test for the Eigenmodel Gibbs sampling algorithm.
"""
import numpy as np
from scipy.stats import norm, probplot
import matplotlib.pyplot as plt

from eigenmodel import LogisticEigenmodel


def demo(seed=None):
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # Create an eigenmodel with N nodes and D-dimensional feature vectors
    N = 5      # Number of nodes
    D = 2       # Dimensionality of the feature space
    p = 0.1    # Baseline average probability of connection
    sigma_mu0 = 1.0     # Variance of the baseline probability
    sigma_F = 1.0     # Variance of the feature space
    sigma_lmbda = 1.0     # Variance of the feature space metric, Lambda
    lmbda = np.ones(D)
    model = LogisticEigenmodel(N=N, D=D, p=p,
                       sigma_mu0=sigma_mu0,
                       sigma_F=sigma_F,
                       sigma_lmbda=sigma_lmbda)

    # Sample a graph from the eigenmodel
    A = model.rvs()

    # Run the Geweke test
    N_samples = 20000
    samples   = []
    for smpl in xrange(N_samples):
        if smpl % 100 == 0:
            print "Iteration ", smpl
        # Resample the model parameters
        model.resample(A)

        # Sample a new graph
        A = model.rvs()

        # Save the sample
        samples.append(model.copy_sample())

    # Check that the samples match the prior
    check_F_samples(model, samples)
    check_mu0_samples(model, samples)
    check_lmbda_samples(model, samples)


def check_F_samples(model, samples):
    mu = 0
    sigma = model.sigma_F

    F_samples = np.array([s.F for s in samples])
    F_mean = F_samples.mean(0)
    F_std   = F_samples.std(0)
    print "Mean F: \n", F_mean, " +- ", F_std


    # Make Q-Q plots
    F_dist = norm(mu, np.sqrt(sigma))
    fig = plt.figure()
    for d in xrange(model.D):
        ax1 = fig.add_subplot(model.D, 2, d*2 + 1)
        probplot(F_samples[:,0,d], dist=F_dist, plot=ax1)

        ax2 = fig.add_subplot(model.D, 2, d*2+2)
        _, bins, _ = ax2.hist(F_samples[:,0,d], 20, normed=True, alpha=0.2)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        ax2.plot(bincenters, F_dist.pdf(bincenters), 'r--', linewidth=1)
        ax2.set_title("F_{0,%d}" % d)
    plt.show()

def check_mu0_samples(model, samples):
    mu = model.mu_mu_0
    sigma = model.sigma_mu0

    mu0_samples = np.array([s.mu_0 for s in samples])
    mu0_mean = mu0_samples.mean(0)
    mu0_std  = mu0_samples.std(0)
    print "Mean mu0: \n", mu0_mean, " +- ", mu0_std


    # Make Q-Q plots
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    mu0_dist = norm(mu, np.sqrt(sigma))
    probplot(mu0_samples, dist=mu0_dist, plot=ax1)

    ax2 = fig.add_subplot(122)
    _, bins, _ = ax2.hist(mu0_samples, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    ax2.plot(bincenters, mu0_dist.pdf(bincenters), 'r--', linewidth=1)
    ax2.set_title("mu0")

    plt.show()


def check_lmbda_samples(model, samples):
    mu = 0
    sigma = model.sigma_lmbda

    lmbda_samples = np.array([s.lmbda for s in samples])
    lmbda_mean = lmbda_samples.mean(0)
    lmbda_std  = lmbda_samples.std(0)
    print "Mean lmbda: \n", lmbda_mean, " +- ", lmbda_std


    # Make Q-Q plots
    lmbda_dist = norm(mu, np.sqrt(sigma))
    fig = plt.figure()
    for d in xrange(model.D):
        ax1 = fig.add_subplot(model.D, 2, d*2 + 1)
        probplot(lmbda_samples[:,d], dist=lmbda_dist, plot=ax1)

        ax2 = fig.add_subplot(model.D, 2, d*2+2)
        _, bins, _ = ax2.hist(lmbda_samples[:,d], 20, normed=True, alpha=0.2)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        ax2.plot(bincenters, lmbda_dist.pdf(bincenters), 'r--', linewidth=1)
        ax2.set_title("lmbda_{%d}" % d)
    plt.show()


demo()