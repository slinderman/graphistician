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
    B = 2       # Weight dimensionality
    p = 0.01    # Baseline average probability of connection
    sigma_F     = 9.0    # Variance of the feature space
    lmbda = np.ones(D)
    mu_lmbda    = 1.0    # Mean of the feature space metric
    sigma_lmbda = 0.1    # Variance of the latent feature space metric
    eigenmodel_args = {"p": p, "sigma_F": sigma_F, "lmbda": lmbda}
    true_model = GaussianWeightedEigenmodel(N=N, D=D, B=B,
                                            eigenmodel_args=eigenmodel_args)
                            # mu_lmbda=mu_lmbda, sigma_lmbda=sigma_lmbda)

    # Sample a graph from the eigenmodel
    network = true_model.rvs()

    # Make another model to fit the data
    test_model = GaussianWeightedEigenmodel(N=N, D=D, B=B,
                                            eigenmodel_args=eigenmodel_args)
                            # mu_lmbda=mu_lmbda, sigma_lmbda=sigma_lmbda)

    # Initialize with the true model settings
    # test_model.init_with_gibbs(true_model)
    test_model.resample_from_mf()

    # Make a figure to plot the true and inferred network
    plt.ion()
    fig     = plt.figure()
    ax_true = fig.add_subplot(1,2,1, aspect="equal")
    true_model.plot(network.A, ax=ax_true)
    ax_test = fig.add_subplot(1,2,2, aspect="equal")
    test_model.plot(network.A, ax=ax_test)

    # Fit with mean field variational inference
    N_iters = 100
    lps     = [test_model.log_probability(network)]
    vlbs    = [test_model.get_vlb() + test_model.expected_log_likelihood(network)]
    for itr in xrange(N_iters):
        # raw_input("Press enter to continue\n")
        print "Iteration ", itr
        test_model.meanfieldupdate(network)
        vlbs.append(test_model.get_vlb() + test_model.expected_log_likelihood(network))
        # vlbs.append(test_model.get_vlb() )

        # Resample from the variational posterior
        test_model.resample_from_mf()
        lps.append(test_model.log_probability((network)))
        print "VLB: ", vlbs[-1]
        print "LP:  ", lps[-1]
        print ""

        # Update the test plot
        if itr % 20 == 0:
            ax_test.cla()
            test_model.plot(network.A, ax=ax_test, color=color, F_true=true_model.F, lmbda_true=true_model.lmbda)
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

    plt.figure()
    plt.plot(np.array(lps))
    plt.xlabel("Iteration")
    plt.ylabel("LP")

    # Plot the inferred weights
    plt.figure()
    plt.errorbar(np.arange(B), test_model.mu_w,
                 yerr=np.sqrt(np.diag(test_model.Sigma_w)),
                 color=color)

    plt.errorbar(np.arange(B), true_model._weight_dist.mf_expected_mu(),
                 yerr=np.sqrt(np.diag(true_model._weight_dist.mf_expected_Sigma())),
                 color='k')

    plt.xlim(-1, B+1)

    plt.show()

# demo(2244520065)
demo()
