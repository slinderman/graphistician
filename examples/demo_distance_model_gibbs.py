"""
Demo of an eigenmodel.
"""
import numpy as np
import matplotlib.pyplot as plt

from distance_model import DistanceModel

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
    N = 30      # Number of nodes
    D = 2       # Dimensionality of the feature space
    true_model = DistanceModel(N=N, D=D)

    # Set the true locations to be on a grid
    # w = 4
    # s = 0.8
    # x = s * (np.arange(N) % w)
    # y = s * (np.arange(N) // w)
    # L = np.hstack((x[:,None], y[:,None]))
    # true_model.L = L

    # Set the true locations to be on a circle
    r = 1.2 + np.arange(N) // (N/2.)
    th = np.linspace(0, 4 * np.pi, N, endpoint=False)
    x = r * np.cos(th)
    y = r * np.sin(th)
    L = np.hstack((x[:,None], y[:,None]))
    true_model.L = L

    # Sample a graph from the eigenmodel
    A = true_model.rvs()

    # Make a figure to plot the true and inferred network
    plt.ion()
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    true_model.plot(A, ax=ax_true)

    test_model = DistanceModel(N=N, D=D)

    # Fit with Gibbs sampling
    N_samples = 1000
    lps       = [test_model.log_probability(A)]
    for smpl in xrange(N_samples):
        print "Iteration ", smpl
        test_model.resample(A)
        lps.append(test_model.log_probability(A))
        print "LP: ", lps[-1]
        print ""


        # Update the test plot
        if smpl % 10 == 0:
            ax_test.cla()
            test_model.plot(A, ax=ax_test, color=color, L_true=true_model.L)
            plt.pause(0.001)

    plt.ioff()
    plt.figure()
    plt.plot(np.array(lps))
    plt.xlabel("Iteration")
    plt.ylabel("LL")
    plt.show()



demo(1234)
