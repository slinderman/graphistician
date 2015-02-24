import numpy as np

from eigenmodel import LogisticEigenmodel
from weights import GaussianWeights

# Create a weighted version with an independent, Gaussian weight model.
# The latent embedding has no bearing on the weight distribution.
class GaussianWeightedEigenmodel():

    def __init__(self, N, D, B,
                 mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None,
                 eigenmodel_args={}):

        self.N = N      # Number of nodes
        self.D = D      # Dimensionality of latent feature space
        self.B = B      # Dimensionality of weights

        # Initialize the graph model
        self.graph_model = LogisticEigenmodel(N, D, **eigenmodel_args)

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

        self.weight_model = GaussianWeights(mu_0=mu_0, kappa_0=kappa_0,
                                            Sigma_0=Sigma_0, nu_0=nu_0)

    # Expose latent variables of the eigenmodel
    @property
    def F(self):
        return self.graph_model.F

    @property
    def mu_0(self):
        return self.graph_model.mu_0

    @property
    def lmbda(self):
        return self.graph_model.lmbda

    @property
    def mu_w(self):
        return self.weight_model.mu

    @property
    def Sigma_w(self):
        return self.weight_model.sigma

    # Extend the basic distribution functions
    def log_prior(self):
        # TODO: Compute the log prior probability for the Normal-Inverse Wishart prior
        return self.graph_model.log_prior()

    def log_likelihood(self, x):
        """
        Compute the log likelihood of
        :param x: A,W tuple
        :return:
        """
        # Extract the adjacency matrix and the nonzero weights
        A,W = x
        W = W[A>0, :]

        ll  = self.graph_model.log_likelihood(A)
        ll += self.weight_model.log_likelihood(W).sum()
        return ll

    def log_probability(self, x):
        lp = self.log_prior()
        lp += self.log_likelihood(x)
        return lp

    def rvs(self, size=[]):
        A = self.graph_model.rvs()
        W = self.weight_model.rvs(size=(self.N, self.N))
        return A, W

    # Extend the Gibbs sampling algorithm
    def resample(self, data=[]):
        """
        Reample given a list of A's and W's
        :param data: (list of A's, list of W's)
        :return:
        """
        A, W = data[0], data[1]

        # Resample the eigenmodel given A
        self.graph_model.resample(A)

        # Resample the weight model given W for which A=1
        W = W[A>0, :]
        self.weight_model.resample(W)

    # Extend the mean field variational inference algorithm
    def meanfieldupdate(self, E_A, E_W, E_WWT):
        """
        Reample given a list of A's and W's
        :param data: (list of A's, list of W's)
        :return:
        """
        # Mean field update the eigenmodel given As
        self.graph_model.meanfieldupdate(E_A)

        # Mean field update the weight model
        # The "weights" of each W correspond to the probability
        # of that connection being nonzero. That is, the weights equal E_A.
        # First, convert E_W and E_WWT to be (N**2, D) and (N**2, D, D), respectively.
        exp_ss_data = [E_W.reshape((self.N**2, self.D)),
                       E_WWT.reshape((self.N**2, self.D, self.D))]
        weights = E_A.reshape((self.N**2,))
        self.weight_model.meanfieldupdate(exp_ss_data=exp_ss_data, weights=weights)

    def get_vlb(self):
        vlb  = self.graph_model.get_vlb()
        vlb += self.weight_model.get_vlb()
        return vlb

    def resample_from_mf(self):
        self.graph_model.resample_from_mf()
        self.weight_model.resample_from_mf()

    def plot(self, A, ax=None, color='k', F_true=None, lmbda_true=None):
        self.graph_model.plot(A, ax, color, F_true, lmbda_true)