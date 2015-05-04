"""
Latent distance model. Each node is embedded in R^D. Probability
of connection is a function of distance in latent space.
"""
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from hips.inference.hmc import hmc

from graphistician.abstractions import GaussianWeightedNetworkDistribution
from graphistician.internals.weights import GaussianWeights, GaussianFixedWeights
from graphistician.internals.utils import logistic


class DistanceModel(object):
    """
    f_n ~ N(0, sigma^2 I)
    A_{n', n} ~ Bern(\sigma(-||f_{n'} - f_{n}||_2^2))
    """
    def __init__(self, N, D, sigma=1.0, mu0=0.0,
                 allow_self_connections=True):
        self.N = N
        self.D = D
        self.sigma = sigma
        self.mu0 = mu0
        self.L = np.sqrt(self.sigma) * np.random.randn(N,D)
        self.allow_self_connections = allow_self_connections

    @property
    def Mu(self):
        Mu = self.mu0 + -((self.L[:,None,:] - self.L[None,:,:])**2).sum(2)

        return Mu

    @property
    def P(self):
        P = logistic(self.Mu)

        if not self.allow_self_connections:
            np.fill_diagonal(P, 1e-32)

        return logistic(self.Mu)

    def log_prior(self):
        """
        Compute the prior probability of F, mu0, and lmbda
        """
        lp  = 0

        # Log prior of F under spherical Gaussian prior
        lp += -0.5 * (self.L * self.L / self.sigma).sum()
        return lp

    def log_likelihood(self, x):
        """
        Compute the log likelihood of a given adjacency matrix
        :param x:
        :return:
        """
        A = x
        assert A.shape == (self.N, self.N)

        P  = self.P
        ll = A * np.log(P) + (1-A) * np.log(1-P)
        if not np.all(np.isfinite(ll)):
            print "P finite? ", np.all(np.isfinite(P))
            import pdb; pdb.set_trace()

        # The graph may contain NaN's to represent missing data
        # Set the log likelihood of NaN's to zero
        ll = np.nan_to_num(ll)

        return ll.sum()

    def log_probability(self, A):
        return self.log_likelihood(A) + self.log_prior()

    def _hmc_log_probability(self, L, mu0, A):
        """
        Compute the log probability as a function of L.
        This allows us to take the gradients wrt L using autograd.
        :param L:
        :param A:
        :return:
        """
        # Compute pairwise distance
        L1 = np.reshape(L,(self.N,1,self.D))
        L2 = np.reshape(L,(1,self.N,self.D))
        D = - np.sum((L1-L2)**2, axis=2)

        # Compute the logit probability
        logit_P = mu0 + D

        # Take the logistic of the negative distance
        # P = 1.0 / (1+np.exp(logit_P))
        P = logistic(logit_P)

        # Compute the log likelihood
        ll = np.sum(A * np.log(P) + (1-A) * np.log(1-P))

        # Log prior of L under spherical Gaussian prior
        lp = -0.5 * np.sum(L * L / self.sigma)

        # Log prior of mu0 under standardGaussian prior
        lp += -0.5 * self.mu0**2

        return ll + lp


    def rvs(self, size=[]):
        """
        Sample a new NxN network with the current distribution parameters

        :param size:
        :return:
        """
        # TODO: Sample the specified number of graphs
        P = self.P
        A = np.random.rand(self.N, self.N) < P

        # TODO: Handle self connections
        # np.fill_diagonal(A, False)

        return A

    def compute_optimal_rotation(self, L_true):
        """
        Find a rotation matrix R such that F_inf * R ~= F_true
        :return:
        """
        # from sklearn.linear_model import LinearRegression
        # assert F_true.shape == (self.N, self.D), "F_true must be NxD"
        # F_inf = self.F
        #
        # # TODO: Scale by lambda in order to get the scaled rotation
        #
        # R = np.zeros((self.D, self.D))
        # for d in xrange(self.D):
        #     lr = LinearRegression(fit_intercept=False)
        #     R[:,d] = lr.fit(F_inf, F_true[:,d]).coef_
        #
        #     # TODO: Normalize this column of the rotation matrix
        #     R[:,d] /= np.linalg.norm(R[:,d])
        #
        from scipy.linalg import orthogonal_procrustes
        R = orthogonal_procrustes(self.L, L_true)[0]
        return R

    def plot(self, A, ax=None, color='k', L_true=None, lmbda_true=None):
        """
        If D==2, plot the embedded nodes and the connections between them

        :param L_true:  If given, rotate the inferred features to match F_true
        :return:
        """
        assert self.D==2, "Can only plot for D==2"

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, aspect="equal")

        # If true locations are given, rotate L to match L_true
        L = self.L
        if L_true is not None:
            R = self.compute_optimal_rotation(L_true)
            L = L.dot(R)

        # Scatter plot the node embeddings
        ax.plot(L[:,0], L[:,1], 's', color=color, markerfacecolor=color, markeredgecolor=color)
        # Plot the edges between nodes
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                if A[n1,n2]:
                    ax.plot([L[n1,0], L[n2,0]],
                            [L[n1,1], L[n2,1]],
                            '-', color=color, lw=1.0)

        # Get extreme feature values
        b = np.amax(abs(L)) + L[:].std() / 2.0

        # Plot grids for origin
        ax.plot([0,0], [-b,b], ':k', lw=0.5)
        ax.plot([-b,b], [0,0], ':k', lw=0.5)

        # Set the limits
        ax.set_xlim([-b,b])
        ax.set_ylim([-b,b])

        # Labels
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        plt.show()

        return ax

    def resample(self, A):
        """
        Resample the parameters of the distribution given the observed graphs.
        :param data:
        :return:
        """
        # Sample the latent positions
        self._resample_L(A)

        # Resample the offset
        self._resample_mu0(A)

    def _resample_L(self, A):
        """
        Resample the locations given A
        :return:
        """
        lp  = lambda L: self._hmc_log_probability(L, self.mu0, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        self.L = hmc(lp, dlp, stepsz, nsteps, self.L.copy(), negative_log_prob=False)

    def _resample_mu0(self, A):
        """
        Resample the locations given A
        :return:
        """
        lp  = lambda mu0: self._hmc_log_probability(self.L, mu0, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        mu0 = hmc(lp, dlp, stepsz, nsteps, np.array(self.mu0), negative_log_prob=False)
        self.mu0 = float(mu0)


class GaussianDistanceModel(GaussianWeightedNetworkDistribution):
    def __init__(self, N, B, D=2, sigma=1.0, mu0=0.0,
                 mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None,
                 allow_self_connections=True):
        self.N = N      # Number of nodes
        self.B = B      # Dimensionality of weights
        self.D = D      # Dimensionality of latent feature space

        # Initialize the graph model
        self._adjacency_dist = DistanceModel(N, D, sigma=sigma, mu0=mu0,
                                             allow_self_connections=allow_self_connections)

        self._weight_dist = GaussianWeights(self.B, mu_0=mu_0, kappa_0=kappa_0,
                                            Sigma_0=Sigma_0, nu_0=nu_0)

    @property
    def adjacency_dist(self):
        return self._adjacency_dist

    @property
    def weight_dist(self):
        return self._weight_dist

    @property
    def allow_self_connections(self):
        return self._adjacency_dist.allow_self_connections

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


    # Extend the basic distribution functions
    def log_prior(self):
        lp = 0
        lp += self.adjacency_dist.log_prior()
        lp += self.weight_dist.log_prior()
        return lp

    def log_likelihood(self, network):
        """
        Compute the log likelihood of
        :param x: A,W tuple
        :return:
        """
        # Extract the adjacency matrix and the nonzero weights
        A, W = network.A, network.W
        W = W[A>0, :]

        ll  = self.adjacency_dist.log_likelihood(A)
        ll += self.weight_dist.log_likelihood(W).sum()
        return ll

    def rvs(self, size=[]):
        A = self.adjacency_dist.rvs()
        W = self.weight_dist.rvs(size=(self.N, self.N))

        return A,W

    # Extend the Gibbs sampling algorithm
    def resample(self, network):
        """
        Reample given a list of A's and W's
        :param data: (list of A's, list of W's)
        :return:
        """
        self.adjacency_dist.resample(network.A)

        # TODO: resample given just W
        self.weight_dist.resample(network)

    # Extend the mean field variational inference algorithm
    def meanfieldupdate(self, network):
        raise NotImplementedError()

    ### Mean field
    def expected_log_likelihood(self, network):
        raise NotImplementedError()

    def get_vlb(self):
        raise NotImplementedError()

    def resample_from_mf(self):
        raise NotImplementedError()

    def svi_step(self, network, minibatchfrac, stepsize):
        raise NotImplementedError()

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



class FixedGaussianDistanceModel(GaussianDistanceModel):
    def __init__(self, N, B, D=2, sigma=1.0, mu0=0.0,
                 mu_W=None, Sigma_W=None):
        self.N = N      # Number of nodes
        self.B = B      # Dimensionality of weights
        self.D = D      # Dimensionality of latent feature space

        # Initialize the graph model
        self._adjacency_dist = DistanceModel(N, D, sigma=sigma, mu0=mu0)

        if mu_W is None:
            mu_W = np.zeros(B)
        if Sigma_W is None:
            Sigma_W = np.eye(B)

        self._weight_dist = GaussianFixedWeights(self.B, mu=mu_W, Sigma=Sigma_W)