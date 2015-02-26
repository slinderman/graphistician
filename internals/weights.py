"""
Prior distribution over weight models that can be combined with the graph models.
"""
import numpy as np

from deps.pybasicbayes.distributions import Gaussian

class GaussianWeights(Gaussian):
    """
    Gaussian weight distribution.
    """
    def __init__(self, B, mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None):
        # Initialize the weight model
        # Set defaults for weight model parameters
        if mu_0 is None:
            mu_0 = np.zeros(B)

        if Sigma_0 is None:
            Sigma_0 = np.eye(B)

        if nu_0 is None:
            nu_0 = B + 2

        if kappa_0 is None:
            kappa_0 = 1.0

        super(GaussianWeights, self).__init__(mu_0=mu_0, sigma_0=Sigma_0,
                                              nu_0=nu_0, kappa_0=kappa_0)

    def log_prior(self):
        # TODO: Compute log prior of Normal-Inverse Wishart
        return 0

    def resample(self, networks=[]):
        if not isinstance(networks, list):
            networks = [networks]

        if len(networks) > 0:
            As = [n.A for n in networks]
            Ws = [n.W for n in networks]
            Ws = np.vstack([W[A==1,:] for A,W in zip(As,Ws)])
        else:
            Ws = []

        # Resample the Normal-inverse Wishart prior over mu and W
        # given W for which A=1
        super(GaussianWeights, self).resample(Ws)

    # Override the mean field updates to allow downstream components to
    # pass in expected statistics of the data.
    def meanfieldupdate(self, network, weights=None):
        """
        Perform mean field update with the expected sufficient statistics of the data
        :param exp_ss_data: a tuple E_x, E_xxT
                            E_x:    NxD matrix of expected weights, W, for each of the N datapoints
                            E_xxT:  NxDxD array of expected WW^T for each of the N datapoints
        :param weights:     how much to weight each datapoint's statistics
        :return:
        """
        E_W, E_WWT  = network.E_W, network.E_WWT

        N = E_W.shape[0]
        D = self.D
        assert E_W.shape == (N,N,D)
        assert E_WWT.shape == (N,N,D,D)

        if weights is not None:
            assert weights.shape == (N,N)
        else:
            weights = np.ones((N,N))

        # Ravel the weights for the Gaussian distribution object
        E_W = E_W.reshape((N**2,D))
        E_WWT = E_WWT.reshape((N**2,D,D))
        weights = weights.reshape((N**2,))

        self.mf_natural_hypparam = \
                self.natural_hypparam + self._get_weighted_statistics(E_W, E_WWT, weights)

    def meanfield_sgdstep(self, network, minibatchfrac, stepsize, weights=None):

        E_W, E_WWT  = network.E_W, network.E_WWT

        N = E_W.shape[0]
        D = self.D
        assert E_W.shape == (N,N,D)
        assert E_WWT.shape == (N,N,D,D)

        if weights is not None:
            assert weights.shape == (N,N)
        else:
            weights = np.ones((N,N))

        # Ravel the weights for the Gaussian distribution object
        E_W = E_W.reshape((N**2,D))
        E_WWT = E_WWT.reshape((N**2,D,D))
        weights = weights.reshape((N**2,))

        self.mf_natural_hypparam = \
                (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                        self.natural_hypparam
                        + 1./minibatchfrac * self._get_weighted_statistics(E_W, E_WWT, weights))

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
    def expected_log_likelihood(self, network):
        from internals.distributions import Gaussian as G

        E_W, E_WWT = network.E_W, network.E_WWT
        N = E_W.shape[0]
        assert E_W.shape == (N,N,self.D)
        assert E_WWT.shape == (N,N,self.D,self.D)

        # E[LN p(W | mu, Sigma)]
        E_mu = self.mf_expected_mu()
        E_mumuT = self.mf_expected_mumuT()
        E_Sigma_inv = self.mf_expected_Sigma_inv()
        E_logdet_Sigma = self.mf_expected_logdet_Sigma()

        ell = 0
        for n_pre in xrange(N):
            for n_post in xrange(N):
                ell += G().negentropy(E_x=E_W[n_pre, n_post, :],
                                      E_xxT=E_WWT[n_pre, n_post, :, :],
                                      E_mu=E_mu,
                                      E_mumuT=E_mumuT,
                                      E_Sigma_inv=E_Sigma_inv,
                                      E_logdet_Sigma=E_logdet_Sigma).sum()

        return ell

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