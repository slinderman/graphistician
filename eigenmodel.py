import numpy as np
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt

from abstractions import AldousHooverNetwork
from deps.pybasicbayes.abstractions import GibbsSampling, MeanField
from utils.utils import sample_truncnorm, expected_truncnorm

class _EigenmodelBase(AldousHooverNetwork):
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

    def __init__(self, N, D, p=0.5, sigma_mu0=1.0, sigma_F=1.0,
                 lmbda=None, mu_lmbda=0, sigma_lmbda=1.0):
        """
        :param N:               The number of nodes we observe.
        :param D:               Dimensionality of the latent feature space.
        :param sigma_mu0:       Variance of the bias, mu_0
        :param sigma_F:               Variance of the feature vectors, u
        :param sigma_lmbda:     Variance of the norm, Lambda
        """
        self.N = N
        self.D = D

        # Initialize distribution parameters
        assert p >= 0 and p <= 1.0, "p must be a baseline probability of connection"
        self.p = p
        self.mu_mu_0 = norm(0,1).ppf(p)
        self.mu_0  = 0

        assert sigma_mu0 >= 0 and sigma_F >= 0 and sigma_lmbda >= 0, "q,r,s must be positive variances"
        self.sigma_mu0 = sigma_mu0
        self.sigma_F = sigma_F
        self.F     = np.zeros((N,D))

        if lmbda is not None:
            assert lmbda.shape == (self.D,)
            self.lmbda_given = True
            self.lmbda = lmbda
        else:
            assert np.isscalar(mu_lmbda) or mu_lmbda.shape == (self.D,)
            self.lmbda_given = False
            self.mu_lmbda = mu_lmbda * np.ones(self.D)
            self.sigma_lmbda = sigma_lmbda
            self.lmbda = self.mu_lmbda.copy()

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
        P = 0.5 * (1 + erf(self.Mu / np.sqrt(2.0)))

        # Clip P so that it is never exactly 1 or 0
        P = np.clip(P, 1e-16, 1-1e-16)
        return P

    def log_prior(self):
        """
        Compute the prior probability of F, mu0, and lmbda
        """
        lp  = 0

        # Log prior of F under spherical Gaussian prior
        lp += -0.5 * (self.F * self.F / self.sigma_F).sum()

        # Add the prior for mu_0 under Gaussian prior
        lp += -0.5 * (self.mu_0 - self.mu_mu_0)**2 / self.sigma_mu0

        # Add prior on lmbda under spherical Gaussian prior
        if not self.lmbda_given:
            lp += -0.5 * (self.lmbda * self.lmbda / self.sigma_lmbda).sum()

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
        np.fill_diagonal(A, False)

        return A


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
        ax.plot(F[:,0], F[:,1], 's', color=color, markerfacecolor=color, markeredgecolor=color)
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


class _GibbsEigenmodel(_EigenmodelBase, GibbsSampling):

    def __init__(self, N, D, **kwargs):
        super(_GibbsEigenmodel, self).__init__(N, D, **kwargs)

        self.resample()

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
        llT = np.outer(self.lmbda, self.lmbda)
        for n in xrange(self.N):
            # Compute the sufficient statistics for fn
            post_prec = 1.0/self.sigma_F * np.eye(self.D)
            post_mean_dot_prec = np.zeros(self.D)

            for Z in Zs:
                # First compute f * Lambda and z-mu0
                # fLambda = self.F * self.lmbda[None,:]
                zcent   = Z - self.mu_0

                for nn in xrange(self.N):
                    if nn == n:
                        continue

                    # post_prec += 2 * np.outer(fLambda[nn,:], fLambda[nn,:])
                    post_prec += 2 * np.outer(self.F[nn,:], self.F[nn,:]) * llT
                    post_mean_dot_prec += zcent[nn,n] * self.F[nn,:] * self.lmbda
                    post_mean_dot_prec += zcent[n,nn] * self.F[nn,:] * self.lmbda

            # Compute the posterior mean and covariance
            post_cov  = np.linalg.inv(post_prec)
            post_mean = post_cov.dot(post_mean_dot_prec)

            # Sample from the posterior
            self.F[n,:] = np.random.multivariate_normal(post_mean, post_cov)

    def _resample_mu_0(self, Zs):
        """
        Sample mu_0 from its Gaussian posterior
        """
        post_prec = 1.0 / self.sigma_mu0
        post_mean_dot_prec = self.mu_mu_0 / self.sigma_mu0

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
        # Compute the sufficient statistics for lmbda
        post_prec = 1.0/self.sigma_lmbda * np.eye(self.D)
        post_mean_dot_prec = self.mu_lmbda / self.sigma_lmbda

        # Add up sufficient statistics for each observed graph
        for Z in Zs:
            # First compute z-mu_0
            zcent = Z - self.mu_0

            # Update the sufficient statistics
            for n1 in xrange(self.N):
                for n2 in xrange(n1+1, self.N):
                    # Update the precision
                    post_prec += 2 * np.outer(self.F[n1,:], self.F[n1,:]) * \
                                     np.outer(self.F[n2,:], self.F[n2,:])

                    # Update the mean dot precision
                    Fn1n2 = self.F[n1,:] * self.F[n2,:]
                    post_mean_dot_prec += zcent[n1,n2] * Fn1n2
                    post_mean_dot_prec += zcent[n2,n1] * Fn1n2

        # Compute the posterior mean and covariance
        post_cov  = np.linalg.inv(post_prec)
        post_mean = post_cov.dot(post_mean_dot_prec)

        # Sample from the posterior
        self.lmbda = np.random.multivariate_normal(post_mean, post_cov)


class _MeanFieldEigenModel(_EigenmodelBase, MeanField):
    def __init__(self, N, D, **kwargs):
        super(_MeanFieldEigenModel, self).__init__(N, D, **kwargs)

        # Initialize mean field parameters
        self.mf_A            = 0.5 * np.ones((N,N))
        self.mf_mu_Z         = np.zeros((N,N))
        self.mf_mu_mu0       = self.mu_mu_0
        self.mf_sigma_mu0    = self.sigma_mu0
        self.mf_mu_F         = np.sqrt(self.sigma_F) * np.random.randn(self.N, self.D)
        self.mf_Sigma_F      = np.tile((self.sigma_F * np.eye(self.D))[None, :, :], [self.N, 1, 1])

        if self.lmbda_given:
            self.mf_mu_lmbda     = self.lmbda.copy()
            self.mf_Sigma_lmbda  = 1e-16 * np.eye(self.D)
        else:
            self.mf_mu_lmbda     = self.mu_lmbda.copy()
            self.mf_Sigma_lmbda  = self.sigma_lmbda * np.eye(self.D)


    def init_with_gibbs(self, gibbs_model):
        self.mf_mu_lmbda = gibbs_model.lmbda.copy()
        self.mf_Sigma_lmbda = 1e-4 * np.eye(self.D)

        self.mf_mu_mu0 = gibbs_model.mu_0
        self.mf_sigma_mu0 = 1e-4

        self.mf_mu_F = gibbs_model.F.copy()
        self.mf_Sigma_F = np.tile((1e-4 * np.eye(self.D))[None, :, :], [self.N,1,1])

    def meanfieldupdate(self, E_A):
        """
        Mean field is a little different than Gibbs. We have to keep around
        variational parameters for the adjacency matrix. Thus, we assume that
        meanfieldupdate is always called with the same, single adjacency matrix.

        Update the mean field variational parameters given the expected
        adjacency matrix, E[A=1].

        :param E_A:     Expected value of the adjacency matrix
        :return:
        """
        # import pdb; pdb.set_trace()
        assert isinstance(E_A, np.ndarray) and E_A.shape == (self.N, self.N) \
               and np.amax(E_A) <= 1.0 and np.amin(E_A) >= 0.0

        # mf_A is just a copy of E_A
        self.mf_A = E_A

        # Update the latent Z's for the adjacency matrix
        self._meanfieldupdate_Z()
        self._meanfieldupdate_mu0()
        self._meanfieldupdate_F()

        if not self.lmbda_given:
            self._meanfieldupdate_lmbda()

    def mf_expected_Z(self):
        """
        Compute the expected value of Z given mf_A and mf_mu_Z
        :return:
        """
        if self.mf_A.dtype == np.bool:
            E_Z  = np.zeros((self.N, self.N))
            E_Z[self.mf_A]  = expected_truncnorm(mu=self.mf_mu_Z, lb=0)[self.mf_A]
            E_Z[~self.mf_A] = expected_truncnorm(mu=self.mf_mu_Z, ub=0)[~self.mf_A]
        else:
            E_Z  = self.mf_A * expected_truncnorm(mu=self.mf_mu_Z, lb=0)
            E_Z += (1-self.mf_A) * expected_truncnorm(mu=self.mf_mu_Z, ub=0)

        assert np.all(np.isfinite(E_Z))
        return E_Z

    def mf_expected_ffT(self, n):
        """
        Compute E[fn * fn^T]
        :param n:
        :return:
        """
        return self.mf_Sigma_F[n,:,:] + np.outer(self.mf_mu_F[n,:], self.mf_mu_F[n,:])

    def mf_expected_llT(self):
        """
        Compute E[lmbda * lmbda^T]
        :param n:
        :return:
        """
        return self.mf_Sigma_lmbda + np.outer(self.mf_mu_lmbda, self.mf_mu_lmbda)

    def _meanfieldupdate_Z(self):
        """
        Update the variational parameters for the adjacency matrix, Z
        :return:
        """
        E_mu0   = self.mf_mu_mu0
        E_F     = self.mf_mu_F
        E_lmbda = self.mf_mu_lmbda

        self.mf_mu_Z = E_mu0 * np.ones((self.N, self.N))
        self.mf_mu_Z += (E_F * E_lmbda[None,:]).dot(E_F.T)

        # TODO: Handle self connections
        np.fill_diagonal(self.mf_mu_Z, E_mu0)

        assert np.all(np.isfinite(self.mf_mu_Z))

    def _meanfieldupdate_mu0(self):
        """
        Sample mu_0 from its Gaussian posterior
        """
        post_prec = 1.0 / self.sigma_mu0
        post_mean_dot_prec = self.mu_mu_0 / self.sigma_mu0

        # Add expected sufficient statistics
        E_Z     = self.mf_expected_Z()
        E_F     = self.mf_mu_F
        E_lmbda = self.mf_mu_lmbda

        post_prec += self.N**2 - self.N

        # Compute the residual of Z after subtracting feature values
        Zcent = E_Z - (E_F * E_lmbda[None, :]).dot(E_F.T)

        # For self connections, we ignore feature values
        np.fill_diagonal(Zcent, 0.0)

        post_mean_dot_prec += Zcent.sum()

        # Set the variational posterior parameters
        self.mf_sigma_mu0 = 1.0 / post_prec
        self.mf_mu_mu0    = self.mf_sigma_mu0 * post_mean_dot_prec

    def _meanfieldupdate_F(self):
        """
        Update the latent features.
        """
        # Sample each feature given the rest
        E_Z     = self.mf_expected_Z()
        E_mu0   = self.mf_mu_mu0
        E_lmbda = self.mf_mu_lmbda

        for n in xrange(self.N):
            # Compute the sufficient statistics for fn
            post_prec = 1.0/self.sigma_F * np.eye(self.D)
            post_mean_dot_prec = np.zeros(self.D)

            # First compute f * Lambda and z-mu0
            zcent   = E_Z - E_mu0

            for nn in xrange(self.N):
                if nn == n:
                    continue

                post_prec += 2 * self.mf_expected_ffT(nn) * self.mf_expected_llT()
                post_mean_dot_prec += zcent[nn,n] * self.mf_mu_F[nn,:] * E_lmbda
                post_mean_dot_prec += zcent[n,nn] * self.mf_mu_F[nn,:] * E_lmbda

            # Set the variational posterior parameters
            self.mf_Sigma_F[n,:,:] = np.linalg.inv(post_prec)
            self.mf_mu_F[n,:]      = self.mf_Sigma_F[n,:,:].dot(post_mean_dot_prec)

        assert np.all(np.isfinite(self.mf_mu_F))
        assert np.all(np.isfinite(self.mf_Sigma_F))

    def _meanfieldupdate_lmbda(self):
        """
        Mean field update for the latent feature space metric
        """
        E_Z     = self.mf_expected_Z()
        E_F     = self.mf_mu_F
        E_mu0   = self.mf_mu_mu0

        # Compute the sufficient statistics for lmbda
        post_prec = 1.0/self.sigma_lmbda * np.eye(self.D)
        post_mean_dot_prec = self.mu_lmbda / self.sigma_lmbda

        # Add up sufficient statistics
        # First center Z
        zcent = E_Z - E_mu0

        # Update the sufficient statistics
        for n1 in xrange(self.N):
            for n2 in xrange(n1+1, self.N):
                # Update the precision
                post_prec += 2 * self.mf_expected_ffT(n1) * self.mf_expected_ffT(n2)

                # Update the mean dot precision
                E_f1f2 = E_F[n1,:] * E_F[n2,:]
                post_mean_dot_prec += zcent[n1,n2] * E_f1f2
                post_mean_dot_prec += zcent[n2,n1] * E_f1f2

        # Compute the posterior mean and covariance
        self.mf_Sigma_lmbda = np.linalg.inv(post_prec)
        self.mf_mu_lmbda    = self.mf_Sigma_lmbda.dot(post_mean_dot_prec)

    def expected_log_likelihood(self,x):
        raise NotImplementedError()

    def get_vlb(self):
        raise NotImplementedError()

    def resample_from_mf(self):
        """
        Resample from the mean field variational posterior
        :return:
        """
        for n in xrange(self.N):
            self.F[n,:] = np.random.multivariate_normal(self.mf_mu_F[n,:], self.mf_Sigma_F[n,:,:])

        self.mu_0 = np.random.normal(self.mf_mu_mu0, np.sqrt(self.mf_sigma_mu0))

        if not self.lmbda_given:
            self.lmbda = np.random.multivariate_normal(self.mf_mu_lmbda, self.mf_Sigma_lmbda)

class Eigenmodel(_GibbsEigenmodel, _MeanFieldEigenModel):
    pass