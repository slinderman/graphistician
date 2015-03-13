import numpy as np
import matplotlib.pyplot as plt

from abstractions import FactorizedWeightedNetworkDistribution, \
    GaussianWeightedNetworkDistribution
from internals.weights import GaussianWeights
from internals.adjacency import BernoulliEdges

# Define an Erdos-Renyi model
class GaussianErdosRenyiFixedSparsity(FactorizedWeightedNetworkDistribution,
                                      GaussianWeightedNetworkDistribution):
    def __init__(self, N, B, p=0.5,
                 mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None):
        super(GaussianErdosRenyiFixedSparsity, self).__init__(N, B)

        self.p = p
        self._adjacency_dist = BernoulliEdges(p)
        self._weight_dist = GaussianWeights(self.B, mu_0=mu_0, kappa_0=kappa_0,
                                            Sigma_0=Sigma_0, nu_0=nu_0)

    @property
    def adjacency_dist(self):
        return self._adjacency_dist

    @property
    def weight_dist(self):
        return self._weight_dist
    @property
    def P(self):
        return self.adjacency_dist.p * np.ones((self.N, self.N))

    @property
    def Mu(self):
        """
        Get the NxNxB array of mean weights
        :return:
        """
        return np.tile(self.weight_dist.mu[None, None, :],
                       [self.N, self.N, 1])

    @property
    def Sigma(self):
        """
        Get the NxNxBxB array of weight covariances
        :return:
        """
        return np.tile(self.weight_dist.sigma[None, None, :, :],
                       [self.N, self.N, 1, 1])

    # Expose network level expectations
    def mf_expected_log_p(self):
        return self.adjacency_dist.mf_expected_log_p() * np.ones((self.N, self.N))

    def mf_expected_log_notp(self):
        return self.adjacency_dist.mf_expected_log_notp() * np.ones((self.N, self.N))

    def mf_expected_mu(self):
        E_mu = self.weight_dist.mf_expected_mu()
        return np.tile(E_mu[None, None, :], [self.N, self.N, 1])

    def mf_expected_mumuT(self):
        E_mumuT = self.weight_dist.mf_expected_mumuT()
        return np.tile(E_mumuT[None, None, :, :], [self.N, self.N, 1, 1])

    def mf_expected_Sigma_inv(self):
        E_Sigma_inv = self.weight_dist.mf_expected_Sigma_inv()
        return np.tile(E_Sigma_inv[None, None, :, :], [self.N, self.N, 1, 1])

    def mf_expected_logdet_Sigma(self):
        E_logdet_Sigma = self.weight_dist.mf_expected_logdet_Sigma()
        return E_logdet_Sigma * np.ones((self.N, self.N))

    def plot(self, network, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect="equal")

        A = network.A
        W = network.W.sum(2)
        Wlim = np.amax(abs(W))
        ax.imshow(A * W, interpolation="none", cmap="RdGy",
                  vmin=-Wlim, vmax=Wlim)


