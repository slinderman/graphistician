import numpy as np

from abstractions import FixedGaussianNetwork, FactorizedWeightedNetworkDistribution, GaussianWeightedNetworkDistribution
from internals.eigenmodel import LogisticEigenmodel
from internals.weights import GaussianWeights

# Create a weighted version with an independent, Gaussian weight model.
# The latent embedding has no bearing on the weight distribution.
class GaussianWeightedEigenmodel(FactorizedWeightedNetworkDistribution, GaussianWeightedNetworkDistribution):

    def __init__(self, N, B, D=2,
                 mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None,
                 **eigenmodel_args):

        super(GaussianWeightedEigenmodel, self).__init__(N, B)

        self.N = N      # Number of nodes
        self.B = B      # Dimensionality of weights
        self.D = D      # Dimensionality of latent feature space

        # Initialize the graph model
        self._adjacency_dist = LogisticEigenmodel(N, D, **eigenmodel_args)

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

        self._weight_dist = GaussianWeights(mu_0=mu_0, kappa_0=kappa_0,
                                            Sigma_0=Sigma_0, nu_0=nu_0)

    @property
    def adjacency_dist(self):
        return self._adjacency_dist

    @property
    def weight_dist(self):
        return self._weight_dist

    # Expose latent variables of the eigenmodel
    @property
    def F(self):
        return self.adjacency_dist.F

    @property
    def P(self):
        return self.adjacency_dist.P

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

    @property
    def mu_0(self):
        return self.adjacency_dist.mu_0

    @property
    def lmbda(self):
        return self.adjacency_dist.lmbda

    @property
    def mu_w(self):
        return self.weight_dist.mu

    @property
    def Sigma_w(self):
        return self.weight_dist.sigma

    # Expose network level expectations
    def mf_expected_log_p(self):
        return self.adjacency_dist.mf_expected_log_p()

    def mf_expected_log_notp(self):
        return self.adjacency_dist.mf_expected_log_notp()

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

    def plot(self, A, ax=None, color='k', F_true=None, lmbda_true=None):
        self.adjacency_dist.plot(A, ax, color, F_true, lmbda_true)