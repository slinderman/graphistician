"""
Demo of an eigenmodel.
"""
import numpy as np
import matplotlib.pyplot as plt

from networks import GaussianWeightedEigenmodel

try:
    from hips.plotting.colormaps import harvard_colors
    color = harvard_colors()[0]
except:
    color = 'b'
# color = 'b'

def demo(seed=None):
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # Create an eigenmodel with N nodes and D-dimensional feature vectors
    N = 10      # Number of nodes
    D = 2       # Dimensionality of the feature space
    p = 0.01    # Baseline average probability of connection
    sigma_F = 3**2    # Scale of the feature space
    lmbda       = np.ones(D)
    mu_lmbda    = 1.0     # Mean of the latent feature space metric
    sigma_lmbda = 0.001   # Variance of the latent feature space metric
    eigenmodel_args = {"p": p, "sigma_F": sigma_F, "lmbda": lmbda}
    B = 2       # Dimensionality of the weights
    true_model = GaussianWeightedEigenmodel(N=N, D=D, B=B,
                                            eigenmodel_args=eigenmodel_args)

    # Sample a graph from the eigenmodel
    # import pdb; pdb.set_trace()
    network = true_model.rvs()

    # Make a figure to plot the true and inferred network
    plt.ion()
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    true_model.plot(network.A, ax=ax_true)

    # Make another model to fit the data
    test_model = GaussianWeightedEigenmodel(N=N, D=D, B=B,
                                            eigenmodel_args=eigenmodel_args)

    # Fit with Gibbs sampling
    N_samples = 1000
    lps       = []
    for smpl in xrange(N_samples):
        print "Iteration ", smpl
        test_model.resample(network)
        lps.append(test_model.log_probability(network))
        print "LP: ", lps[-1]
        print ""


        # Update the test plot
        if smpl % 20 == 0:
            ax_test.cla()
            test_model.plot(network.A, ax=ax_test, color=color, F_true=true_model.F, lmbda_true=true_model.lmbda)
            plt.pause(0.001)

    plt.ioff()
    # Plot the log likelihood as a function of iteration
    plt.figure()
    plt.plot(np.array(lps))
    plt.xlabel("Iteration")
    plt.ylabel("LL")

    # Plot the inferred weights
    plt.figure()
    plt.errorbar(np.arange(B), test_model.mu_w,
                 yerr=np.sqrt(np.diag(test_model.Sigma_w)),
                 color=color)

    plt.errorbar(np.arange(B), true_model.mu_w,
                 yerr=np.sqrt(np.diag(true_model.Sigma_w)),
                 color='k')

    plt.xlim(-1, B+1)

    plt.show()

demo()
