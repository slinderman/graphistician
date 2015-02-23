import numpy as np
from scipy.stats import truncnorm, norm
from graphistician.utils.distributions import TruncatedScalarGaussian, ScalarGaussian
from graphistician.utils.utils import normal_cdf

def test_truncated_scalar_gaussian_lb():
    tn0_test = TruncatedScalarGaussian(lb=0)
    tn0_true = truncnorm(0, np.Inf)

    print "E[TN(0,inf)]:\t", tn0_test.expected_x()
    print "E[TN(0,inf)]:\t", tn0_true.mean()
    assert np.allclose(tn0_test.expected_x(), tn0_true.mean())

    print "Var[TN(0,inf)]:\t", tn0_test.variance_x()

def test_truncated_scalar_gaussian():
    lb = 0
    ub = 10
    mu = 3
    sigma = 1
    tn0_test = TruncatedScalarGaussian(mu=mu, sigmasq=np.sqrt(sigma), lb=lb, ub=ub)
    tn0_true = truncnorm((lb-mu)/sigma, (ub-mu)/sigma, loc=mu)

    print "E[TN(0,10)]:\t", tn0_test.expected_x()
    print "E[TN(0,10)]:\t", tn0_true.mean()
    assert np.allclose(tn0_test.expected_x(), tn0_true.mean())

    print "Var[TN(0,10)]:\t", tn0_test.variance_x()
    print "Var[TN(0,10)]:\t", tn0_true.var()
    assert np.allclose(tn0_test.variance_x(), tn0_true.var())

    print "E[-LN{TN(0,10)}]:\t", -tn0_test.negentropy()
    print "E[-LN{TN(0,10)}]:\t", tn0_true.entropy()
    assert np.allclose(-tn0_test.negentropy(), tn0_true.entropy())

def test_truncated_gaussian_entropy(mu=0.0, sigma=1.0):
    """
    Consider
    Z ~ N(mu,1)
    versus
    A ~ Bern(p)   for    p = 1-Phi(-mu)
    Z ~ A * TN(mu, 1, 0, Inf)  +  (1-A) * TN(mu, 1, -Inf, 0)

    We have simply separated the Gaussian sampling into two steps:
    1. Sample the sign of Z
    2. Sample Z from a truncated normal.

    Thus, E_Z[LN p(Z)] should equal E_A[ E_{Z|A} [LN p (Z | A)]]
    using iterated expectations.
    :return:
    """
    p  = 1.0 - normal_cdf(0, mu, sigma)
    print "p:   ", p

    # Compute the entropy of the scalar Gaussian
    H0 = ScalarGaussian(mu=mu, sigmasq=sigma**2).negentropy()

    # Compute the entropy of the two stage process
    # H1 = p * TruncatedScalarGaussian(lb=0, mu=mu, sigmasq=sigma).negentropy() + \
    #      (1-p) * TruncatedScalarGaussian(ub=0, mu=mu, sigmasq=sigma).negentropy()

    TN1 = TruncatedScalarGaussian(lb=0, mu=mu, sigmasq=sigma)
    EZ1 = TN1.expected_x()
    EZ1sq = TN1.expected_xsq()
    H1 = p * ScalarGaussian(mu=mu, sigmasq=sigma).negentropy(E_x=EZ1, E_xsq=EZ1sq)

    TN0 = TruncatedScalarGaussian(ub=0, mu=mu, sigmasq=sigma)
    EZ0 = TN0.expected_x()
    EZ0sq = TN0.expected_xsq()
    H1 += (1-p) * ScalarGaussian(mu=mu, sigmasq=sigma).negentropy(E_x=EZ0, E_xsq=EZ0sq)

    H2 = -norm(loc=mu, scale=sigma).entropy()
    # H3 = -p * truncnorm((0-mu)/sigma, np.inf, loc=mu).entropy() + \
    #      -(1-p) * truncnorm(-np.inf, (0-mu)/sigma, loc=mu).entropy()

    print "H0:  ", H0
    print "H1:  ", H1
    print "H2:  ", H2
    # print "H3:  ", H3

    # assert np.allclose(H0, H1)

# test_truncated_scalar_gaussian()
test_truncated_gaussian_entropy(mu=1.0)
