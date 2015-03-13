"""
Latent distance model. Each node is embedded in R^D. Probability
of connection is a function of distance in latent space.
"""
import numpy as np

import matplotlib.pyplot as plt

from abstractions import NetworkDistribution
from utils import logistic

class DistanceModel(object):
    """
    f_n ~ N(0, sigma^2 I)
    A_{n', n} ~ Bern(\sigma(-||f_{n'} - f_{n}||_2^2))
    """
    def __init__(self, N, D, sigma=1):
        self.N = N
        self.D = D
        self.sigma = sigma

        self.L = np.sqrt(self.sigma) * np.random.randn(N,D)

    @property
    def Mu(self):
        Mu = -((self.L[:,None,:] - self.L[None,:,:])**2).sum(2)

        # TODO: Handle self connections
        np.fill_diagonal(Mu, 0)

        return Mu

    @property
    def P(self):
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
            R = self.compute_optimal_rotation(L_true, lmbda_true)
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
        ax.set_xlabel('Latent Feature 1')
        ax.set_ylabel('Latent Feature 2')
        plt.show()

        return ax

    def resample(self, data=[]):
        """
        Resample the parameters of the distribution given the observed graphs.
        :param data:
        :return:
        """
        if not isinstance(data, list):
            data = [data]

        # Sample auxiliary variables for each graph in data
        Zs = self._resample_Z(data)

        # Sample per-node features given Z, mu_0, and lmbda
        self._resample_F(Zs)

        # Sample the bias
        self._resample_mu_0(Zs)

        # Sample the metric of the latent feature space
        if not self.lmbda_given:
            self._resample_lmbda(Zs)