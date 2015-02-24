"""
Prior distribution over weight models that can be combined with the graph models.
"""
import numpy as np

from deps.pybasicbayes.distributions import Gaussian

class GaussianWeights(Gaussian):
    """
    Gaussian weight distribution.
    """
    def __init__(self, mu_0, Sigma_0, nu_0, kappa_0):
        super(GaussianWeights, self).__init__(mu_0=mu_0, sigma_0=Sigma_0,
                                              nu_0=nu_0, kappa_0=kappa_0)

    # Override the mean field updates to allow downstream components to
    # pass in expected statistics of the data.
    def meanfieldupdate(self, exp_ss_data, weights=None):
        """
        Perform mean field update with the expected sufficient statistics of the data
        :param exp_ss_data: a tuple E_x, E_xxT
                            E_x:    NxD matrix of expected weights, W, for each of the N datapoints
                            E_xxT:  NxDxD array of expected WW^T for each of the N datapoints
        :param weights:     how much to weight each datapoint's statistics
        :return:
        """
        E_x, E_xxT = exp_ss_data
        N = E_x.shape[0]
        D = self.D
        assert E_x.shape == (N,D)
        assert E_xxT.shape == (N,D,D)

        if weights is not None:
            assert weights.shape == (N,)
        else:
            weights = np.ones(N)

        self.mf_natural_hypparam = \
                self.natural_hypparam + self._get_weighted_statistics(E_x, E_xxT, weights)

    def meanfield_sgdstep(self, exp_ss_data, minibatchfrac, stepsize, weights=None):

        E_x, E_xxT = exp_ss_data
        N = E_x.shape[0]
        D = self.D
        assert E_x.shape == (N,D)
        assert E_xxT.shape == (N,D,D)

        if weights is not None:
            assert weights.shape == (N,)
        else:
            weights = np.ones(N)

        self.mf_natural_hypparam = \
                (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                        self.natural_hypparam
                        + 1./minibatchfrac * self._get_weighted_statistics(E_x, E_xxT, weights))

    def _get_weighted_statistics(self, E_x, E_xxT, weights):
        """
        Compute the weighted sum of the sufficient statistics
        :param E_x:
        :param E_xxT:
        :param weights:
        :return:
        """
        D = self.D
        out = np.zeros((D+2,D+2))
        out[:D,:D] = (weights[:, None, None] * E_xxT).sum(0)
        out[-2,:D] = out[:D,-2] = (weights[:,None] * E_x).sum(0)
        out[-2,-2] = out[-1,-1] = weights.sum()
        return out

    # Expose mean field expectations
    def mf_expected_mu(self):
        return self.mu_mf

    def mf_expected_Sigma(self):
        # Expectation of W^{-1} (S, nu) = S / (nu - D - 1)
        return self.sigma_mf / (self.nu_mf - self.D - 1)

    def mf_expected_mumuT(self):
        # E[mu mu^T] = E[Sigma] + E[mu]E[mu]^T
        E_Sigma = self.mf_expected_Sigma()
        E_mu    = self.mu_mf
        return E_Sigma + np.outer(E_mu, E_mu)

    def mf_expected_Sigma_inv(self):
        # Expectation of W(S^{-1}, nu) = nu * S^{-1}
        return self.nu_mf * np.linalg.inv(self.sigma_mf)

    def mf_expected_logdet_Sigma(self):
        return -self._loglmbdatilde()

    def resample_from_mf(self):
        self._resample_from_mf()

class GammaWeights:

    def __init__(self):
        raise NotImplementedError()