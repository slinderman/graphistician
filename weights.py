"""
Prior distribution over weight models that can be combined with the graph models.
"""
import numpy as np

from graphistician.deps.pybasicbayes.distributions import Gaussian

class GaussianWeights(Gaussian):
    """
    Gaussian weight distribution.
    """
    def __init__(self, mu_0, Sigma_0, nu_0, kappa_0):
        super(Gaussian, self).__init__(mu_0=mu_0, sigma_0=Sigma_0, nu_0=nu_0, kappa_0=kappa_0)

    # Expose mean field expectations
    def mf_expected_mu(self):
        return self.mu_mf

    def mf_expected_mumuT(self):
        # E[mu mu^T] = E[Sigma] + E[mu]E[mu]^T
        E_Sigma = self.sigma_mf / self.nu_mf / self.kappa_mf
        E_mu    = self.mu_mf
        return E_Sigma + np.outer(E_mu, E_mu)

    def mf_expected_Sigma_inv(self):
        return self.nu_mf * np.linalg.inv(self.sigma_mf)

    def mf_expected_logdet_Sigma(self):
        return -self._loglmbdatilde()


class GammaWeights:
    raise NotImplementedError()