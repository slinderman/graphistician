"""
Demo of an eigenmodel.
"""
import numpy as np
import matplotlib.pyplot as plt

from graphistician.stochastic_block_model import GaussianStochasticBlockModel


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
    N = 100      # Number of nodes
    C = 5       # Dimensionality of the feature space
    sbm_args = {"C": C}
    B = 2       # Dimensionality of the weights
    true_model = GaussianStochasticBlockModel(N, B, **sbm_args)

    # Sample a graph from the eigenmodel
    # import pdb; pdb.set_trace()
    network = true_model.rvs()

    # Make a figure to plot the true and inferred network
    plt.ion()
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    true_model.plot(network, ax=ax_true)

    # Make another model to fit the data
    test_model = GaussianStochasticBlockModel(N, B, **sbm_args)
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    test_model.plot(network, ax=ax_test)

    plt.pause(0.001)

    # Fit with Gibbs sampling
    N_samples = 100
    lps       = [test_model.log_probability(network)]
    for smpl in xrange(N_samples):
        print "Iteration ", smpl
        test_model.resample(network)
        lps.append(test_model.log_probability(network))
        print "LP: ", lps[-1]
        print ""

        # Update the test plot
        if smpl % 1 == 0:
            ax_test.cla()
            test_model.plot(network, ax=ax_test)
            plt.pause(0.001)

    plt.ioff()
    # Plot the log likelihood as a function of iteration
    plt.figure()
    plt.plot(np.array(lps))
    plt.xlabel("Iteration")
    plt.ylabel("LL")

    plt.show()

demo()
