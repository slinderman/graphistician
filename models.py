import numpy as np
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt

from abstractions import AldousHooverNetwork
from deps.pybasicbayes.abstractions import GibbsSampling
from utils.utils import sample_truncnorm

class Eigenmodel(AldousHooverNetwork, GibbsSampling):
    """
    Eigenmodel for random graphs, as defined in Hoff, 2008.

    A_{i,j} = I[z_{i,j} >= 0]
    z_{i,j} ~ N(mu_{0} + f_i^T \Lambda f_j, 1)
    Lambda  = diag(lambda)
    mu_{0}  ~ N(0, q)
    f_{i}   ~ N(0, rI)
    lambda  ~ N(0, sI)

    lambda and f_{i} are vectors in R^D.

    Probability of a connection i->j is proportional to the dot product
    of the vectors f_i and f_j under the norm Lambda.  Hence we can think of
    the f's as feature vectors, and the probability of connection as
    proportional to feature similarity.

    We have to decide where the data-local variables will reside. Here,
    we let the Eigenmodel own the data-local variables. This implies that
    the distribution object applies to a fixed size network. Alternatively,
    we could augment the "data," i.e. the adjacency matrix, with node-specific
    and edge-specific random variables and let the Eigenmodel only own the
    global random variables. This alternative seems more cumbersome for use
    in other hierarchical models.
    """

    def __init__(self, N, D, p=0.5, q=1.0, r=1.0, s=1.0):
        """
        :param N:     The number of nodes we observe.
        :param D:     Dimensionality of the latent feature space.
        :param q:     Variance of the bias, mu_0
        :param r:     Variance of the feature vectors, u
        :param s:     Variance of the norm, Lambda
        """
        self.N = N
        self.D = D

        assert p >= 0 and p <= 1.0, "p must be a baseline probability of connection"
        self.p = p
        self.mu_mu_0 = norm(0,1).ppf(p)

        assert q >= 0 and r >= 0 and s >= 0, "q,r,s must be positive variances"
        self.q = q
        self.r = r
        self.s = s

        # Initialize distribution parameters
        self.mu_0  = 0
        self.F     = np.zeros((N,D))
        self.lmbda = np.ones((D,))

        # Resample from the prior
        self.resample()

    @property
    def Mu(self):
        """
        Compute the mean of the Gaussian auxiliary variables
        """
        return self.mu_0 + (self.F * self.lmbda[None,:]).dot(self.F.T)

    @property
    def P(self):
        """
        Compute the probability of each edge.
        """
        return 0.5 * (1 + erf(self.Mu / np.sqrt(2.0)))

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

        # The graph may contain NaN's to represent missing data
        # Set the log likelihood of NaN's to zero
        ll = np.nan_to_num(ll)

        return ll.sum()

    def rvs(self, size=[]):
        """
        Sample a new NxN network with the current distribution parameters

        :param size:
        :return:
        """
        # TODO: Sample the specified number of graphs
        P = self.P
        A = np.random.rand(self.N, self.N) < P
        return A

    # def logpr_f(self,f,theta):
    #     """
    #     The features of an eigenmodel are the feature vectors. They
    #     have a spherical Gaussian prior:
    #         f_i ~ N(0, rI)
    #     """
    #     N,D = f.shape
    #     assert D == self.D
    #
    #     return -0.5 * (f * f / self.r).sum()
    #
    # def logpr_theta(self, theta):
    #     """
    #     The globals are mu_0 and lmbda
    #     """
    #     mu_0, lmbda, Z = theta
    #     assert np.isscalar(mu_0)
    #     assert lmbda.shape == (self.D,)
    #     lp = 0.0
    #
    #     # Add the prior for mu_0
    #     lp += -0.5 * mu_0**2 / self.q
    #
    #     # Add prior on lmbda
    #     lp += -0.5 * (lmbda * lmbda / self.s).sum()
    #
    #     return lp

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
        self._resample_lmbda(Zs)

    def _resample_Z(self, data=[]):
        """
        Resample auxiliary Gaussian variables for each entry in each graph.
        The Z's follow a truncated Gaussian conditional distribution.
        """
        N  = self.N
        Mu = self.Mu
        Zs = []
        for A in data:
            # Get the upper and lower bounds
            lb = -np.Inf * np.ones((N,N))
            lb[A==1] = 0

            ub = np.Inf * np.ones((N,N))
            ub[A==0] = 0

            Zs.append(sample_truncnorm(mu=Mu, lb=lb, ub=ub))
            assert np.all(np.isfinite(Zs[-1]))

        return Zs

    def _resample_F(self, Zs):
        """
        Sample new block assignments given the parameters.

        :param beta: The weight of the log likelihood
        """
        # Sample each feature given the rest
        for n in xrange(self.N):
            # Compute the sufficient statistics for fn
            post_prec = 1.0/self.r * np.eye(self.D)
            post_mean_dot_prec = np.zeros(self.D)

            for Z in Zs:
                # First compute f * Lambda and z-mu0
                fLambda = self.F * self.lmbda[None,:]
                zcent   = Z - self.mu_0

                for nn in xrange(self.N):
                    if nn == n:
                        continue

                    post_prec += 2 * np.outer(fLambda[nn,:], fLambda[nn,:])
                    post_mean_dot_prec += zcent[nn,n] * fLambda[nn,:]
                    post_mean_dot_prec += zcent[n,nn] * fLambda[nn,:]

            # Compute the posterior mean and covariance
            post_cov  = np.linalg.inv(post_prec)
            post_mean = post_cov.dot(post_mean_dot_prec)

            # Sample from the posterior
            self.F[n,:] = np.random.multivariate_normal(post_mean, post_cov)

    def _resample_mu_0(self, Zs):
        """
        Sample mu_0 from its Gaussian posterior
        """
        post_prec = 1.0 / self.q
        post_mean_dot_prec = self.mu_mu_0 / self.q

        # Add sufficient statistics from each Z
        for Z in Zs:
            post_prec += self.N**2 - self.N

            # Compute the residual of Z after subtracting feature values
            Zcent = Z - (self.F * self.lmbda[None, :]).dot(self.F.T)

            # For self connections, we ignore feature values
            np.fill_diagonal(Zcent, 0.0)

            post_mean_dot_prec += Zcent.sum()

        # Compute the posterior mean (assuming prior mean is zero)
        post_var  = 1.0 / post_prec
        post_mean = post_var * post_mean_dot_prec

        # Sample from the posterior
        self.mu_0 =  np.random.normal(post_mean, np.sqrt(post_var))

    def _resample_lmbda(self, Zs):
        """
        Sample lambda from its multivariate normal posterior
        """
        # First compute the Hadamard product of each pair of feature vectors
        FdotF = self.F[:,None,:] * self.F[None,:,:]
        assert FdotF.shape == (self.N, self.N, self.D)
        # Zero out the diagonal elements since we ignore self connections
        for n in xrange(self.N):
            FdotF[n,n,:] = 0

        # Compute the sufficient statistics for lmbda
        post_prec = 1.0/self.s * np.eye(self.D)
        post_mean_dot_prec = np.zeros(self.D)

        # Add up sufficient statistics for each observed graph
        for Z in Zs:
            # First compute z-mu_0
            zcent   = Z - self.mu_0

            # Update the sufficient statistics
            for n1 in xrange(self.N):
                for n2 in xrange(self.N):
                    if n1 == n2:
                        continue

                    # Update the precision
                    post_prec += np.outer(FdotF[n1,n2,:], FdotF[n1,n2,:])

                    # Update the mean dot precision
                    post_mean_dot_prec += zcent[n1,n2] * FdotF[n1,n2,:]

        # Compute the posterior mean and covariance
        post_cov  = np.linalg.inv(post_prec)
        post_mean = post_cov.dot(post_mean_dot_prec)

        # Sample from the posterior
        self.lmbda = np.random.multivariate_normal(post_mean, post_cov)

    def compute_optimal_rotation(self, F_true, lmbda_true):
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
        R = orthogonal_procrustes(self.F / np.sqrt(abs(self.lmbda[None, :])),
                                  F_true / np.sqrt(abs(lmbda_true[None,:])))[0]
        return R

    def plot(self, A, ax=None, color='k', F_true=None, lmbda_true=None):
        """
        If D==2, plot the embedded nodes and the connections between them

        :param F_true:  If given, rotate the inferred features to match F_true
        :return:
        """
        assert self.D==2, "Can only plot for D==2"

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, aspect="equal")

        # If F_true is given, rotate F to match F_true
        F = self.F / np.sqrt(abs(self.lmbda[None, :]))
        if F_true is not None:
            R = self.compute_optimal_rotation(F_true, lmbda_true)
            F = F.dot(R)

        # Scatter plot the node embeddings
        ax.plot(F[:,0], F[:,1], 's', color=color, markerfacecolor=color)
        # Plot the edges between nodes
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                if A[n1,n2]:
                    ax.plot([F[n1,0], F[n2,0]],
                            [F[n1,1], F[n2,1]],
                            '-', color=color, lw=1.0)

        # Get extreme feature values
        b = np.amax(abs(F)) + F[:].std() / 2.0

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