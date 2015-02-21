"""
Demo of an eigenmodel.
"""
import numpy as np
import matplotlib.pyplot as plt

from models import Eigenmodel

def demo(seed=None):
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # Create an eigenmodel with N nodes and D-dimensional feature vectors
    N = 10      # Number of nodes
    D = 2       # Dimensionality of the feature space
    p = 0.01    # Baseline average probability of connection
    r = 3**2      # Scale of the feature space
    true_model = Eigenmodel(N=N, D=D, p=p, r=r)

    # Override the latent feature metric
    true_model.lmbda = np.ones((2,))

    # Sample a graph from the eigenmodel
    A = true_model.rvs()

    # Make a figure to plot the true and inferred network
    plt.ion()
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    true_model.plot(A, ax=ax_true)

    # Make another model to fit the data
    test_model = Eigenmodel(N=N, D=D, p=p, r=r)
    # Override the latent feature metric
    # test_model.lmbda = 0.001 * np.ones((2,))

    # Fit with Gibbs sampling
    N_samples = 200
    lls       = []
    for smpl in xrange(N_samples):
        print "Iteration ", smpl
        test_model.resample(A)
        lls.append(test_model.log_likelihood(A))
        print "LL: ", lls[-1]
        print ""


        # Update the test plot
        ax_test.cla()
        test_model.plot(A, ax=ax_test, color='b', F_true=true_model.F, lmbda_true=true_model.lmbda)
        plt.pause(0.001)

    plt.ioff()
    plt.figure()
    plt.plot(np.array(lls))
    plt.xlabel("Iteration")
    plt.ylabel("LL")
    plt.show()

demo()
