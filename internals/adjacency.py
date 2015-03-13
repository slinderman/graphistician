"""
Super simple adjacency models. We either have a Bernoulli model
with fixed probability or a beta-Bernoulli model.
"""
import numpy as np
from scipy.special import psi

from pybasicbayes.distributions import Distribution

class BernoulliEdges(Distribution):
    """
    Bernoulli edge model with fixed probability
    """
    def __init__(self, p):
        assert p > 0 and p < 1
        self.p = p
        self.ln_p = np.log(self.p)
        self.ln_notp = np.log(1-self.p)

    def rvs(self, size=[]):
        return np.random.rand(*size) < self.p

    def log_prior(self):
        return 0

    def log_likelihood(self, x):
        assert np.all(np.bitwise_or(x==0, x==1))
        return (x * self.ln_p + (1-x) * self.ln_notp).sum()

    def resample(self, edges):
        pass

    def meanfieldupdate(self, As, weights=None):
        pass

    def mf_expected_log_p(self):
        return self.ln_p

    def mf_expected_log_notp(self):
        return self.ln_notp

    def expected_log_likelihood(self, x):
        return self.log_likelihood(x)

    def get_vlb(self):
        return 0

    def resample_from_mf(self):
        pass

    def svi_step(self, network, minibatchfrac, stepsize, weights=None):
        pass


class BetaBernoulliEdges(Distribution):
    def __init__(self, tau1, tau0):
        assert tau1 > 0 and tau0 > 0
        self.tau1 = tau1
        self.tau0 = tau0

    def rvs(self, size=[]):
        return np.random.rand(*size) < self.p

    def log_prior(self):
        return

    def log_likelihood(self, x):
        assert np.all(x==0 | x==1)
        return x * np.log(self.p) + (1-x) * np.log(1-self.p)

    def resample(self, As=[]):
        """
        Resample p given observations of the weights
        """
        n_conns = sum([A.sum() for A in As])
        n_noconns = sum([A.size for A in As]) - n_conns

        tau1 = self.tau1 + n_conns
        tau0 = self.tau0 + n_noconns

        self.p = np.random.beta(tau1, tau0)

    def meanfieldupdate(self, As, weights=None, stepsize=1.0):
        # TODO: Use network as input
        tau1_hat = self.tau1 + (pc1c2 * E_A).sum()
        tau0_hat = self.tau0 + (pc1c2 * E_notA).sum()

        self.mf_tau1[c1,c2] = (1.0 - stepsize) * self.mf_tau1[c1,c2] + stepsize * tau1_hat
        self.mf_tau0[c1,c2] = (1.0 - stepsize) * self.mf_tau0[c1,c2] + stepsize * tau0_hat


    def expected_log_likelihood(self, x):
        return self.log_likelihood(x)

    def mf_expected_p(self):
        """
        Compute the expected probability of a connection, averaging over c
        :return:
        """
        E_p = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_p += pc1c2 * self.mf_tau1[c1,c2] / (self.mf_tau0[c1,c2] + self.mf_tau1[c1,c2])

        if not self.allow_self_connections:
            np.fill_diagonal(E_p, 0.0)

        return E_p

    def mf_expected_notp(self):
        """
        Compute the expected probability of NO connection, averaging over c
        :return:
        """
        return 1.0 - self.mf_expected_p()

    def mf_expected_log_p(self):
        """
        Compute the expected log probability of a connection, averaging over c
        :return:
        """
        E_ln_p = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_p += pc1c2 * Beta(self.mf_tau1[c1,c2], self.mf_tau0[c1,c2]).expected_log_p()

        if not self.allow_self_connections:
            np.fill_diagonal(E_ln_p, -np.inf)

        return E_ln_p

    def mf_expected_log_notp(self):
        """
        Compute the expected log probability of NO connection, averaging over c
        :return:
        """
        E_ln_notp = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_notp += pc1c2 * Beta(self.mf_tau1[c1,c2],
                                          self.mf_tau0[c1,c2]).expected_log_notp()

        if not self.allow_self_connections:
            np.fill_diagonal(E_ln_notp, 0.0)

        return E_ln_notp

    def get_vlb(self):
        # Get the VLB of the connection probability matrix
        vlb = 0
        # Add the cross entropy of p(p | tau1, tau0)
        vlb += Beta(self.tau1, self.tau0).\
            negentropy(E_ln_p=(psi(self.mf_tau1) - psi(self.mf_tau0 + self.mf_tau1)),
                       E_ln_notp=(psi(self.mf_tau0) - psi(self.mf_tau0 + self.mf_tau1))).sum()

        # Subtract the negative entropy of q(p)
        vlb -= Beta(self.mf_tau1, self.mf_tau0).negentropy().sum()

        return vlb
