import numpy as np
from scipy.stats import truncnorm
from graphistician.utils.distributions import TruncatedScalarGaussian

def test_truncated_scalar_gaussian_lb():
    tn0_test = TruncatedScalarGaussian(lb=0)
    tn0_true = truncnorm(0, np.Inf)

    print "E[TN(0,inf)]:\t", tn0_test.expected_x()
    print "E[TN(0,inf)]:\t", tn0_true.mean()
    assert np.allclose(tn0_test.expected_x(), tn0_true.mean())

    print "Var[TN(0,inf)]:\t", tn0_test.variance_x()

def test_truncated_scalar_gaussian():
    tn0_test = TruncatedScalarGaussian(lb=0, ub=10)
    tn0_true = truncnorm(0, 10)

    print "E[TN(0,10)]:\t", tn0_test.expected_x()
    print "E[TN(0,10)]:\t", tn0_true.mean()
    assert np.allclose(tn0_test.expected_x(), tn0_true.mean())

    print "Var[TN(0,10)]:\t", tn0_test.variance_x()
    print "Var[TN(0,10)]:\t", tn0_true.var()
    assert np.allclose(tn0_test.variance_x(), tn0_true.var())

    print "E[-LN{TN(0,10)}]:\t", -tn0_test.negentropy()
    print "E[-LN{TN(0,10)}]:\t", tn0_true.entropy()
    assert np.allclose(-tn0_test.negentropy(), tn0_true.entropy())


test_truncated_scalar_gaussian()