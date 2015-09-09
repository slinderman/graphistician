"""
Demo of a model comparison test where we synthesize a network
and then try to fit it with a variety of network models.
We compare models with heldout predictive likelihood.
"""
import numpy as np
import matplotlib.pyplot as plt
from graphistician.adjacency import \
    LatentDistanceAdjacencyDistribution, \
    SBMAdjacencyDistribution, \
    BetaBernoulliAdjacencyDistribution


from pybasicbayes.util.profiling import show_line_stats
PROFILING=True

def demo(seed=None):
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # Create an latent distance model with N nodes and D-dimensional locations
    N = 30      # Number of nodes
    N_test = 1  # Number of nodes to hold out for testing
    D = 2       # Dimensionality of the feature space
    true_model = LatentDistanceAdjacencyDistribution(N=N+N_test, dim=D)

    # Sample a graph from the eigenmodel
    Afull = true_model.rvs()
    Atrain = Afull[:N,:N]
    Atest_row = Afull[N:,:].ravel()
    Atest_col = Afull[:,N:].ravel()

    # Make a figure to plot the true and inferred network
    test_models = [
        LatentDistanceAdjacencyDistribution(N=N, dim=D),
        SBMAdjacencyDistribution(N=N, C=4),
        BetaBernoulliAdjacencyDistribution(N=N)
    ]

    # Fit each model with Gibbs sampling
    results = []
    N_samples = 1000
    for test_model in test_models:
        lps       = [test_model.log_probability(Atrain)]
        plls      = [test_model.approx_predictive_ll(Atest_row, Atest_col)]
        for smpl in xrange(N_samples):
            print "Iteration ", smpl
            test_model.resample(Atrain)
            lps.append(test_model.log_probability(Atrain))
            plls.append(test_model.approx_predictive_ll(Atest_row, Atest_col))
            print "LP:  ", lps[-1]
            print "PLL: ", plls[-1]
            print ""

        results.append((test_model.__class__, lps, plls))


    colors = ['b', 'r', 'g']
    plt.figure()
    for col, (cls, lps, plls) in zip(colors, results):
        plt.subplot(121)
        plt.plot(lps, color=col)
        plt.xlabel("Iteration")
        plt.ylabel("LP")

        plt.subplot(122)
        plt.plot(plls, color=col)
        plt.xlabel("Iteration")
        plt.ylabel("PLL")

    plt.show()

demo(1234)
show_line_stats()