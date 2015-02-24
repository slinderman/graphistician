import numpy as np

from utils import normal_cdf, normal_pdf

class Bernoulli:
    #TODO: Subclass Discrete distribution
    def __init__(self, p=0.5):
        assert np.all(p >= 0) and np.all(p <= 1.0)
        self.p = p

    def log_probability(self, x):
        """
        Log probability of x given p
        :param x:
        :return:
        """
        lp = x * np.log(self.p) + (1-x) * np.log(1.0-self.p)
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.p

    def expected_notx(self):
        return 1 - self.p

    def negentropy(self, E_x=None, E_notx=None, E_ln_p=None, E_ln_notp=None):
        """
        Compute the entropy of the Bernoulli distribution.
        :param E_x:         If given, use this in place of expectation wrt p
        :param E_notx:      If given, use this in place of expectation wrt p
        :param E_ln_p:      If given, use this in place of expectation wrt p
        :param E_ln_notp:   If given, use this in place of expectation wrt p
        :return: E[ ln p(x | p)]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_notx is None:
            E_notx = self.expected_notx()

        if E_ln_p is None:
            E_ln_p = np.log(self.p)

        if E_ln_notp is None:
            E_ln_notp = np.log(1.0 - self.p)

        if E_x.dtype == np.bool:
            H = np.zeros_like(E_x)
            H[E_x] = E_ln_p[E_x]
            H[E_notx] = E_ln_notp[E_notx]
        else:
            H = E_x * E_ln_p + E_notx * E_ln_notp
        # H = np.nan_to_num(H)
        return H


class ScalarGaussian:

    def __init__(self, mu=0.0, sigmasq=1.0):
        assert np.all(sigmasq) >= 0
        self.mu = mu
        self.sigmasq = sigmasq

    def log_probability(self, x):
        """
        Log probability of x given mu, sigmasq
        :param x:
        :return:
        """
        lp = -0.5*np.log(2*np.pi*self.sigmasq) -1.0/(2*self.sigmasq) * (x-self.mu)**2
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.mu

    def expected_xsq(self):
        return self.sigmasq + self.mu**2


    def negentropy(self, E_x=None, E_xsq=None, E_mu=None, E_musq=None, E_sigmasq_inv=None, E_ln_sigmasq=None):
        """
        Compute the negative entropy of the Gaussian distribution
        :return: E[ ln p(x | mu, sigmasq)] = E[-0.5*log(2*pi*sigmasq) - 0.5/sigmasq * (x-mu)**2]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_xsq is None:
            E_xsq = self.expected_xsq()

        if E_mu is None:
            E_mu = self.mu

        if E_musq is None:
            E_musq = self.mu**2

        if E_sigmasq_inv is None:
            E_sigmasq_inv = 1.0/self.sigmasq

        if E_ln_sigmasq is None:
            E_ln_sigmasq = np.log(self.sigmasq)

        H  = -0.5 * np.log(2*np.pi)
        H += -0.5 * E_ln_sigmasq
        H += -0.5 * E_sigmasq_inv * E_xsq
        H += E_sigmasq_inv * E_x * E_mu
        H += -0.5 * E_sigmasq_inv * E_musq
        return H



class TruncatedScalarGaussian:

    def __init__(self, mu=0.0, sigmasq=1.0, lb=-np.Inf, ub=np.Inf):
        assert np.all(sigmasq) >= 0

        # Broadcast arrays to be of the same shape
        self.mu, self.sigmasq, self.lb, self.ub \
            = np.broadcast_arrays(mu, sigmasq, lb, ub)

        # Precompute the normalizers
        self.zlb = (self.lb-self.mu) / np.sqrt(self.sigmasq)
        self.zub = (self.ub-self.mu) / np.sqrt(self.sigmasq)
        self.Z   = normal_cdf(self.zub) - normal_cdf(self.zlb)

        # Make sure Z is at least epsilon
        # self.Z = np.clip(self.Z, 1e-32, 1.0)

    def log_probability(self, x):
        """
        Log probability of x given mu, sigmasq
        :param x:
        :return:
        """
        assert np.all(x >= self.lb) and np.all(x <= self.ub)

        # Center x
        z = (x-self.mu) / np.sqrt(self.sigmasq)
        # Log prob of the normalization constant
        lp = -0.5 * np.log(self.sigmasq) - np.log(self.Z)
        # Log prob of the standardized density
        lp += -0.5*np.log(2*np.pi) -0.5*z**2
        return lp

    def expected_x(self):
        return self.mu + \
               np.sqrt(self.sigmasq) * (normal_pdf(self.zlb) - normal_pdf(self.zub))/self.Z

    def variance_x(self):
        trm1 = (np.nan_to_num(self.zlb) * normal_pdf(self.zlb) -
                np.nan_to_num(self.zub) * normal_pdf(self.zub)) / self.Z
        trm2 = ((normal_pdf(self.zlb) - normal_pdf(self.zub)) / self.Z)**2
        return self.sigmasq * (1 + trm1 - trm2)

    def expected_xsq(self):
        return self.variance_x() + self.expected_x()**2

    def negentropy(self, E_x=None, E_xsq=None,
                   E_mu=None, E_musq=None,
                   E_sigmasq_inv=None, E_ln_sigmasq=None,
                   E_ln_Z=None):
        """
        Compute the negative entropy of the Gaussian distribution
        :return: E[ ln p(x | mu, sigmasq)]
                 = E[-0.5*log(2*pi*sigmasq) - 0.5/sigmasq * (x-mu)**2 - log(Z)]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_xsq is None:
            E_xsq = self.expected_xsq()

        if E_mu is None:
            E_mu = self.mu

        if E_musq is None:
            E_musq = self.mu**2

        if E_sigmasq_inv is None:
            E_sigmasq_inv = 1.0/self.sigmasq

        if E_ln_sigmasq is None:
            E_ln_sigmasq = np.log(self.sigmasq)

        if E_ln_Z is None:
            E_ln_Z = np.log(self.Z)

        H  = -0.5 * np.log(2*np.pi)
        H += -0.5 * E_ln_sigmasq
        H += -0.5 * E_sigmasq_inv * E_xsq
        H += E_sigmasq_inv * E_x * E_mu
        H += -0.5 * E_sigmasq_inv * E_musq
        H -= E_ln_Z

        return H


class Gaussian:

    def __init__(self, mu=np.zeros(1), Sigma=np.eye(1)):
        assert mu.ndim == 1, "Mu must be a 1D vector"
        self.mu = mu
        self.D  = mu.shape[0]

        assert Sigma.shape == (self.D, self.D), "Sigma must be a DxD covariance matrix"
        self.Sigma = Sigma
        self.logdet_Sigma = np.linalg.slogdet(self.Sigma)[1]

    def log_probability(self, x):
        """
        Log probability of x given mu, sigmasq
        :param x:
        :return:
        """
        z = x-self.mu
        lp = -0.5*self.D*np.log(2*np.pi) -0.5*self.D*self.logdet_Sigma \
             -0.5 * z.T.dot(np.linalg.solve(self.Sigma, z))
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.mu

    def expected_xxT(self):
        return self.Sigma + np.outer(self.mu, self.mu)


    def negentropy(self, E_x=None, E_xxT=None, E_mu=None, E_mumuT=None, E_Sigma_inv=None, E_logdet_Sigma=None):
        """
        Compute the negative entropy of the Gaussian distribution
        :return: E[ ln p(x | mu, sigmasq)] = E[-0.5*log(2*pi) -0.5*E[log |Sigma|] - 0.5 * (x-mu)^T Sigma^{-1} (x-mu)]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_xxT is None:
            E_xxT = self.expected_xxT()

        if E_mu is None:
            E_mu = self.mu

        if E_mumuT is None:
            E_mumuT = np.outer(self.mu, self.mu)

        if E_Sigma_inv is None:
            E_Sigma_inv = np.linalg.inv(self.Sigma)

        if E_logdet_Sigma is None:
            E_logdet_Sigma = np.linalg.slogdet(self.Sigma)[1]

        H  = -0.5 * np.log(2*np.pi)
        H += -0.5 * E_logdet_Sigma
        # TODO: Replace trace with something more efficient
        H += -0.5 * np.trace(E_Sigma_inv.dot(E_xxT))
        H += E_x.T.dot(E_Sigma_inv).dot(E_mu)
        H += -0.5 * np.trace(E_Sigma_inv.dot(E_mumuT))

        return H