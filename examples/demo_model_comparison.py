"""
Demo of a model comparison test where we synthesize a network
and then try to fit it with a variety of network models.
We compare models with heldout predictive likelihood.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

from pybasicbayes.util.text import progprint_xrange

from graphistician.adjacency import \
    LatentDistanceAdjacencyDistribution, \
    SBMAdjacencyDistribution, \
    BetaBernoulliAdjacencyDistribution

from graphistician.weights import \
    NIWGaussianWeightDistribution, \
    SBMGaussianWeightDistribution

from graphistician.networks import FactorizedNetworkDistribution

seed = 1234
# seed = np.random.randint(2**32)

# Create an latent distance model with N nodes and D-dimensional locations
N = 30      # Number of nodes
N_test = 1  # Number of nodes to hold out for testing
B = 1       # Dimensionality of the weights
D = 2       # Dimensionality of the feature space

true_model = FactorizedNetworkDistribution(
    N+N_test,
    LatentDistanceAdjacencyDistribution, {},
    NIWGaussianWeightDistribution, {})

# Sample a graph from the eigenmodel
Afull, Wfull = true_model.rvs()
Atrain= Afull[:N,:N]
Wtrain= Wfull[:N,:N,:]
Atest_row = Afull[N:,:].ravel()
Wtest_row = Wfull[N:,:,:].reshape((N+N_test,1))
Atest_col = Afull[:,N:].ravel()
Wtest_col = Wfull[:,N:,:].reshape((N+N_test,1))

# Make a figure to plot the true and inferred network
adj_models = [
    LatentDistanceAdjacencyDistribution,
    SBMAdjacencyDistribution,
    BetaBernoulliAdjacencyDistribution,
]

weight_models = [
    NIWGaussianWeightDistribution,
    SBMGaussianWeightDistribution
]

# Fit each model with Gibbs sampling
results = []
N_samples = 1000
for adj_model, weight_model in itertools.product(adj_models, weight_models):
    test_model = FactorizedNetworkDistribution(N, adj_model, {}, weight_model, {})
    lps       = [test_model.log_probability((Atrain, Wtrain))]
    plls      = [test_model.approx_predictive_ll(Atest_row, Atest_col, Wtest_row, Wtest_col)]

    print "A: ", adj_model.__name__
    print "W: ", weight_model.__name__

    for smpl in progprint_xrange(N_samples):
        test_model.resample((Atrain, Wtrain))
        lps.append(test_model.log_probability((Atrain, Wtrain)))
        plls.append(test_model.approx_predictive_ll(Atest_row, Atest_col, Wtest_row, Wtest_col))

    results.append((adj_model.__name__ + "-" + weight_model.__name__, lps, plls))


colors = ['b', 'r', 'g', 'y', 'm', 'k']
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

