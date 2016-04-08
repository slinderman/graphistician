"""
Demo of an eigenmodel.
"""
import numpy as np
import matplotlib.pyplot as plt

from graphistician import GaussianErdosRenyiFixedSparsity


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
    B = 2       # Dimensionality of the weights
    true_model = GaussianErdosRenyiFixedSparsity(N, B)

    # Sample a graph from the eigenmodel
    network = true_model.rvs()

    # Make another model to fit the data
    test_model = GaussianErdosRenyiFixedSparsity(N, B)

    # Fit with Gibbs sampling
    N_iters = 20
    lps     = [test_model.log_probability(network)]
    for itr in xrange(N_iters):
        print "Iteration ", itr
        test_model.meanfieldupdate(network)
        test_model.resample_from_mf()

        lps.append(test_model.log_probability(network))
        print "LP: ", lps[-1]
        print ""

    plt.ioff()
    # Plot the log likelihood as a function of iteration
    plt.figure()
    plt.plot(np.array(lps))
    plt.xlabel("Iteration")
    plt.ylabel("LL")

    plt.show()

demo()
