
import numpy as np

from abstractions import WeightDistribution
from pybasicbayes.distributions import Gaussian

class GaussianWeightDistribution(WeightDistribution):
    def __init__(self, N, B, mu, sigma):
        super(GaussianWeightDistribution, self).__init__(N)
        self.B = B

        assert mu.shape == (B,)
        self.mu = mu

        assert sigma.shape == (B,B)
        self.sigma = sigma

        self._gaussian = Gaussian(mu, sigma)

    @property
    def Mu(self):
        raise NotImplementedError()

    @property
    def Sigma(self):
        raise NotImplementedError()

    def log_prior(self):
        return 0

    def resample(self, (A,W)):
        pass


class NIWGaussianWeightDistribution(WeightDistribution):
    """
    Gaussian weight distribution with a normal inverse-Wishart prior.
    """
    def __init__(self, N, B, mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None):
        super(NIWGaussianWeightDistribution, self).__init__(N)
        self.B = B


        if mu_0 is None:
            mu_0 = np.zeros(B)

        if Sigma_0 is None:
            Sigma_0 = np.eye(B)

        if nu_0 is None:
            nu_0 = B + 2

        if kappa_0 is None:
            kappa_0 = 1.0

        self._gaussian = Gaussian(mu_0=mu_0, sigma_0=Sigma_0,
                                  nu_0=nu_0, kappa_0=kappa_0)

    @property
    def Mu(self):
        raise NotImplementedError()

    @property
    def Sigma(self):
        raise NotImplementedError()

    def log_prior(self):
        # TODO: Compute log prior of Normal-Inverse Wishart
        return 0

    def resample(self, (A,W)):
        # Resample the Normal-inverse Wishart prior over mu and W
        # given W for which A=1
        self._gaussian.resample(W[A==1])


class LowRankGaussianWeightDistribution(WeightDistribution):
    """
    Low rank weight matrix (i.e. BPMF from Minh and Salakhutidnov)
    """
    pass

class SBMGaussianWeightDistribution(WeightDistribution):
    """
    Stochastic block model with Gaussian weights.
    """
    pass