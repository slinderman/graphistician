import numpy as np
from graphistician.weights import GaussianWeights

def test_mf_expectations():
    """
    Test the mean field expectations for the Gaussian weights
    """
    D = 2
    mu_0 = np.zeros(D)
    Sigma_0 = np.eye(D)
    nu_0 = 5
    kappa_0 = 1
    w = GaussianWeights(mu_0=mu_0, Sigma_0=Sigma_0, kappa_0=kappa_0, nu_0=nu_0)


    # Compare empirical and analytic expectations
    mu_smpls = []
    mumuT_smpls = []
    Sigma_inv_smpls = []
    Sigma_smpls = []
    logdet_Sigma_smpls = []

    N_samples = 10000
    for smpl in xrange(N_samples):
        if smpl % 1000 == 0:
            print "Sample ", smpl
        w._resample_from_mf()
        mu_smpls.append(w.mu)
        mumuT_smpls.append(np.outer(w.mu, w.mu))
        Sigma_smpls.append(w.sigma)
        Sigma_inv_smpls.append(np.linalg.inv(w.sigma))
        logdet_Sigma_smpls.append(np.log(np.linalg.det(w.sigma)))

    mu_smpls = np.array(mu_smpls)
    mumuT_smpls = np.array(mumuT_smpls)
    Sigma_smpls = np.array(Sigma_smpls)
    Sigma_inv_smpls = np.array(Sigma_inv_smpls)
    logdet_Sigma_smpls = np.array(logdet_Sigma_smpls)

    # Compare means
    print "Analytic  E[mu]:\t", w.mf_expected_mu()
    print "Empirical E[mu]:\t", mu_smpls.mean(0)
    print ""

    print "Analytic  E[mumuT]:\t", w.mf_expected_mumuT()
    print "Empirical E[mumuT]:\t", mumuT_smpls.mean(0)
    print ""

    print "Analytic  E[S]:\t", w.mf_expected_Sigma()
    print "Empirical E[S]:\t", Sigma_smpls.mean(0)
    print ""

    print "Analytic  E[S^{-1}]:\t", w.mf_expected_Sigma_inv()
    print "Empirical E[S^{-1}]:\t", Sigma_inv_smpls.mean(0)
    print ""

    print "Analytic  E[log |S|]:\t", w.mf_expected_logdet_Sigma()
    print "Empirical E[log |S|]:\t", logdet_Sigma_smpls.mean(0)
    print ""

test_mf_expectations()
