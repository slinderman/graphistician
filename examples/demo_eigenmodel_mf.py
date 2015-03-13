"""
Demo of an eigenmodel.
"""
import numpy as np
import matplotlib.pyplot as plt

from graphistician.eigenmodel import ProbitEigenmodel
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
    sigma_F     = 9.0    # Variance of the feature space
    lmbda = np.ones(D)
    mu_lmbda    = 1.0    # Mean of the feature space metric
    sigma_lmbda = 0.1    # Variance of the latent feature space metric
    true_model = ProbitEigenmodel(N=N, D=D, p=p, sigma_F=sigma_F,
                            lmbda=lmbda)
                            # mu_lmbda=mu_lmbda, sigma_lmbda=sigma_lmbda)

    # Override the latent feature metric
    true_model.lmbda = np.ones((2,))

    # Sample a graph from the eigenmodel
    A = true_model.rvs()

    # Make another model to fit the data
    test_model = ProbitEigenmodel(N=N, D=D, p=p, sigma_F=sigma_F,
                            lmbda=lmbda)
                            # mu_lmbda=mu_lmbda, sigma_lmbda=sigma_lmbda)

    # Initialize with the true model settings
    # test_model.init_with_gibbs(true_model)
    test_model.resample_from_mf()

    # Make a figure to plot the true and inferred network
    plt.ion()
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    true_model.plot(A, ax=ax_true)
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    test_model.plot(A, ax=ax_test)

    # Fit with mean field variational inference
    N_iters = 1000
    lps     = []
    vlbs    = []
    for itr in xrange(N_iters):
        # raw_input("Press enter to continue\n")
        print "Iteration ", itr
        test_model.meanfieldupdate(A)
        vlbs.append(test_model.get_vlb())

        # Resample from the variational posterior
        test_model.resample_from_mf()
        lps.append(test_model.log_probability(A))
        print "VLB: ", vlbs[-1]
        # print "LP:  ", lps[-1]
        print ""

        # Update the test plot
        if itr % 20 == 0:
            ax_test.cla()
            test_model.plot(A, ax=ax_test, color=color, L_true=true_model.F, lmbda_true=true_model.lmbda)
            plt.pause(0.001)

    # Analyze the VLBs
    vlbs = np.array(vlbs)
    # vlbs[abs(vlbs) > 1e8] = np.nan
    # finite = np.where(np.isfinite(vlbs))[0]
    # finite_vlbs = vlbs[finite]
    # vlbs_increasing = np.all(np.diff(finite_vlbs) >= -1e-3)
    # print "VLBs increasing? ", vlbs_increasing


    plt.ioff()
    plt.figure()
    # plt.plot(finite, finite_vlbs)
    plt.plot(vlbs)
    plt.xlabel("Iteration")
    plt.ylabel("VLB")

    # plt.figure()
    # plt.plot(np.array(lps))
    # plt.xlabel("Iteration")
    # plt.ylabel("LL")
    plt.show()

# demo(2244520065)
demo(11223344)
