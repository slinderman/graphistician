"""
Demo of an eigenmodel.
"""
import numpy as np

import matplotlib.pyplot as plt

from internals.distance_model import DistanceModel

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
    N = 20      # Number of nodes
    D = 2       # Dimensionality of the feature space
    true_model = DistanceModel(N=N, D=D)

    # Sample a graph from the eigenmodel
    A = true_model.rvs()

    # Make a figure to plot the true and inferred network
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    true_model.plot(A, ax=ax_true)



demo()
