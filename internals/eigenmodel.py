import abc
import numpy as np
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt

from abstractions import DirectedNetwork
from deps.pybasicbayes.abstractions import GibbsSampling, MeanField
from utils.utils import sample_truncnorm, expected_truncnorm, normal_cdf, logistic
from utils.distributions import ScalarGaussian, TruncatedScalarGaussian, Gaussian, Bernoulli

from pypolyagamma import pgdrawv, PyRNG


class _EigenmodelBase(object):
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
        # return self.mu_0 + (self.F * self.lmbda[None,:]).dot(self.F.T)
        Mu = self.mu_0 + (self.F * self.lmbda[None,:]).dot(self.F.T)
        np.fill_diagonal(Mu, self.mu_0)
        return Mu

    @abc.abstractproperty
    def P(self):
        """
        Compute the probability of each edge.
        """
        raise NotImplementedError()

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
        # np.fill_diagonal(A, False)

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


class _ProbitEigenmodelBase(_EigenmodelBase):
    """
    An eigenmodel with a probit link function.
    """
    @property
    def P(self):
        """
        Compute the probability of each edge.
        """
        P = 0.5 * (1 + erf(self.Mu / np.sqrt(2.0)))

        # Clip P so that it is never exactly 1 or 0
        P = np.clip(P, 1e-16, 1-1e-16)
        return P


class _GibbsProbitEigenmodel(_ProbitEigenmodelBase, GibbsSampling):

    def __init__(self, N, D, **kwargs):
        super(_GibbsProbitEigenmodel, self).__init__(N, D, **kwargs)

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


class _MeanFieldProbitEigenModel(_ProbitEigenmodelBase, MeanField):
    def __init__(self, N, D, **kwargs):
        super(_MeanFieldProbitEigenModel, self).__init__(N, D, **kwargs)

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

    def mf_expected_log_p(self):
        """
        Compute the expected log probability of a connection under Z
        :return:
        """
        mu = self.mf_mu_Z
        # return np.nan_to_num(np.log(1.0 - normal_cdf(0, mu=mu, sigma=1.0)))
        return np.log(1.0 - normal_cdf(0, mu=mu, sigma=1.0))

    def mf_expected_log_notp(self):
        """
        Compute the expected log probability of no connection under Z
        :return:
        """
        mu = self.mf_mu_Z
        return np.log(normal_cdf(0, mu=mu, sigma=1.0))

    def mf_expected_log_p_mc(self):
        N_samples = 100
        E_mu = self.mf_expected_mu()[None, :, :]
        std_mu = np.sqrt(self.mf_variance_mu()[None, :, :])
        mus = E_mu + std_mu * np.random.randn(N_samples, self.N, self.N)
        # ps = 1.0 - normal_cdf(0, mu=mus, sigma=1.0)
        # log_ps = np.log(ps)
        # log_notps = np.log(1-ps)

        # Approximate log(1-p) for p~= 1
        u = normal_cdf(0, mu=mus, sigma=1.0)
        log_ps = np.log1p(-u)
        log_notps = np.log1p(-1+u)

        return log_ps.mean(0), log_notps.mean(0)

    def mf_expected_mu(self):
        """
        E[mu] = E[mu0 + f^T \Lambda f]
              = E[mu0] + E[f]^T E[\Lambda] E[f]
        :return:
        """
        E_mu = self.mf_mu_mu0 + (self.mf_mu_F * self.mf_mu_lmbda[None,:]).dot(self.mf_mu_F.T)

        # Override the diagonal
        np.fill_diagonal(E_mu, self.mf_mu_mu0)
        return E_mu

    def mf_expected_musq(self):
        """
        E[mu_{n1n2}^2] = E[(mu0 + f_{n1}^T \Lambda f_{n2})^2]
                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + E[(f_{n1}^T \Lambda f_{n2})^2]

                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + E[(\sum_{k} lmbda_k f_{n1, k} f_{n2, k})^2]

                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + E[\sum_{k1} \sum_{k2} f_{n1, k1} \lmbda_{k1} f_{n2,k1}
                                            x    f_{n1, k2} \lmbda_{k2} f_{n2, k2}]

                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + sum(E[f_{n1} f_{n1}^T]  E[\lmbda \lmbda^T] E[f_{n2} f_{n2}^T])

            for n1 \neq n2
        :return:
        """
        E_musq = np.zeros((self.N, self.N))
        E_musq += self.mf_expected_mu0sq()
        E_musq += 2*self.mf_mu_mu0 * (self.mf_mu_F * self.mf_mu_lmbda[None, :]).dot(self.mf_mu_F.T)
        llT = self.mf_expected_llT()
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                E_musq[n1,n2] += np.sum(self.mf_expected_ffT(n1)
                                        * llT
                                        * self.mf_expected_ffT(n2))

        # Override the diagonal
        np.fill_diagonal(E_musq, self.mf_expected_mu0sq())

        return E_musq

    def mf_variance_mu(self):
        return self.mf_expected_musq() - self.mf_expected_mu()**2

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

        # assert np.all(np.isfinite(E_Z))
        return E_Z

    def mf_expected_Zsq(self):
        if self.mf_A.dtype == np.bool:
            E_Zsq  = np.zeros((self.N, self.N))
            E_Zsq[self.mf_A]  = TruncatedScalarGaussian(mu=self.mf_mu_Z, lb=0).expected_xsq()[self.mf_A]
            E_Zsq[~self.mf_A] = TruncatedScalarGaussian(mu=self.mf_mu_Z, ub=0).expected_xsq()[~self.mf_A]
        else:
            E_Zsq  = self.mf_A * TruncatedScalarGaussian(mu=self.mf_mu_Z, lb=0).expected_xsq()
            E_Zsq += (1-self.mf_A) * TruncatedScalarGaussian(mu=self.mf_mu_Z, ub=0).expected_xsq()

        # assert np.all(np.isfinite(E_Zsq))
        # if np.amax(abs(E_Zsq)) > 1e8:
        #     import pdb; pdb.set_trace()
        return E_Zsq

    def mf_expected_mu0sq(self):
        return self.mf_sigma_mu0 + self.mf_mu_mu0**2

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
        """
        Compute the expected log likelihood of a graph x=A.
        :param x:
        :return:
        """
        A = x
        assert A.shape == (self.N, self.N) and np.all(np.bitwise_or(A==0, A==1))
        log_p    = self.mf_expected_log_p()
        log_notp = self.mf_expected_log_notp()
        return (A * log_p + (1-A) * log_notp).sum()

    def get_vlb(self):
        """
        Compute the variational lower bound.
        :return:
        """
        vlb = 0

        # p(A=1) [E_{Z | A} [ln N(z | mu, sigma=1)],
        # where z ~ TN(mu, 1, lb=0)
        E_mu = self.mf_expected_mu()
        E_musq = self.mf_expected_musq()
        qZ1 = TruncatedScalarGaussian(lb=0, mu=self.mf_mu_Z, sigmasq=1.0)
        tmp = ScalarGaussian().negentropy(E_x=qZ1.expected_x(),
                                          E_xsq=qZ1.expected_xsq(),
                                          E_mu=E_mu,
                                          E_musq=E_musq)
        vlb += (self.mf_A * tmp).sum()

        # p(A=0) [E_{Z | A} [ln N(z | mu, sigma=1)],
        # where z ~ TN(mu, 1, ub=0)
        qZ0 = TruncatedScalarGaussian(ub=0, mu=self.mf_mu_Z, sigmasq=1.0)
        tmp = ScalarGaussian().negentropy(E_x=qZ0.expected_x(),
                                          E_xsq=qZ0.expected_xsq(),
                                          E_mu=E_mu,
                                          E_musq=E_musq)

        vlb += ((1-self.mf_A) * tmp).sum()

        # -E[ln q(z | A, mf_mu_z, 1)] for A=1
        tmp = TruncatedScalarGaussian(lb=0,
                                      mu=self.mf_mu_Z,
                                      sigmasq=1.0).negentropy()
        vlb -= ( self.mf_A * tmp).sum()

        # -E[ln q(z | A, mf_mu_z, 1)] for A=0
        tmp = TruncatedScalarGaussian(ub=0,
                                      mu=self.mf_mu_Z,
                                      sigmasq=1.0).negentropy()

        vlb -= ((1-self.mf_A) * tmp).sum()

        # E[ln p(mu0 | mu_{mu0}, sigma_{mu0})]
        vlb += ScalarGaussian(mu=self.mu_mu_0, sigmasq=self.sigma_mu0)\
               .negentropy(E_x=self.mf_mu_mu0,
                           E_xsq=self.mf_expected_mu0sq())

        # -E[ln q(mu0 | mf_mu_mu0, mf_sigma_mu0)]
        vlb -= ScalarGaussian(mu=self.mf_mu_mu0, sigmasq=self.mf_sigma_mu0).negentropy()

        # E[ln p(F | 0, sigma_{F})]
        # -E[ln q(F | mf_mu_F, mf_sigma_F)]
        for n in xrange(self.N):
            vlb += Gaussian(mu=np.zeros(self.D),
                            Sigma=self.sigma_F *np.eye(self.D))\
                .negentropy(E_x=self.mf_mu_F[n,:], E_xxT=self.mf_expected_ffT(n))

            vlb -= Gaussian(mu=self.mf_mu_F[n,:],
                            Sigma=self.mf_Sigma_F[n,:,:]).negentropy()

        # E[ln p(lmbda | mu_lmbda, sigma_{lmbda})]
        if not self.lmbda_given:
            vlb += Gaussian(mu=self.mu_lmbda,
                            Sigma=self.sigma_lmbda * np.eye(self.D))\
                .negentropy(E_x=self.mf_mu_lmbda, E_xxT=self.mf_expected_llT())

            # -E[ln q(lmbda | mf_mu_lmbda, mf_sigma_lmbda)]
            vlb -= Gaussian(mu=self.mf_mu_lmbda, Sigma=self.mf_Sigma_lmbda).negentropy()

        return vlb

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


class ProbitEigenmodel(_GibbsProbitEigenmodel, _MeanFieldProbitEigenModel):
    pass


class _LogisticEigenmodelBase(_EigenmodelBase):
    """
    Eigenmodel for random graphs, as defined in Hoff, 2008.

    A_{i,j} ~ Bern(\sigma(z_{i,j}))
    z_{i,j} = mu_{0} + f_i^T \Lambda f_j
    Lambda  = diag(lambda)
    mu_{0}  ~ N(0, q)
    f_{i}   ~ N(0, rI)
    lambda  ~ N(0, sI)

    lambda and f_{i} are vectors in R^D.

    Probability of a connection i->j is a function of the dot product
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

    @property
    def P(self):
        """
        Compute the probability of each edge.
        """
        P = logistic(self.Mu)
        # Clip P so that it is never exactly 1 or 0
        P = np.clip(P, 1e-16, 1-1e-16)
        return P

    def kappa(self, A):
        """
        kappa = a - b/2, which for a logistic Bernoulli model is
              = 1 - A/2
        :param A:
        :return:
        """
        return A - 0.5

class _GibbsLogisticEigenmodel(_LogisticEigenmodelBase, GibbsSampling):
    def __init__(self, N, D, **kwargs):
        super(_GibbsLogisticEigenmodel, self).__init__(N, D, **kwargs)

        self.rng = PyRNG()

        # DEBUG:
        self.F = np.sqrt(self.sigma_F) * np.random.randn(self.N, self.D)
        self.mu_0 = self.mu_mu_0 + np.sqrt(self.sigma_mu0) * np.random.randn()

        self.resample()

    def resample(self, networks=[]):
        """
        Resample the parameters of the distribution given the observed graphs.
        :param networks:
        :return:
        """
        if not isinstance(networks, list):
            networks = [networks]
        assert all([isinstance(n, DirectedNetwork) for n in networks])
        As = [network.A for network in networks]

        # Sample auxiliary variables for each graph in data
        Omegas = self._resample_Omega(As)

        # Sample per-node features given Z, mu_0, and lmbda
        self._resample_F(As, Omegas)

        # Sample the bias
        self._resample_mu_0(As, Omegas)

        # Sample the metric of the latent feature space
        if not self.lmbda_given:
            self._resample_lmbda(As, Omegas)

    def _resample_Omega(self, As=[]):
        """
        Sample auxiliary Polya-gamma variables for each adjacency matrix
        :param As:
        :return:
        """
        Omegas = []
        for A in As:
            tmp = np.empty(A.size, dtype=np.float)
            pgdrawv(np.ones(A.size, dtype=np.int32),
                    self.Mu.ravel("C"),
                    tmp,
                    self.rng)
            Omega = tmp.reshape((self.N, self.N), order="C")
            Omegas.append(Omega)
        return Omegas

    def _resample_F(self, As, Omegas):
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

            for A,O in zip(As, Omegas):
                # First compute f * Lambda and z-mu0
                # fLambda = self.F * self.lmbda[None,:]
                zcent   = self.kappa(A)/O - self.mu_0

                for nn in xrange(self.N):
                    if nn == n:
                        continue

                    post_prec += np.outer(self.F[nn,:], self.F[nn,:]) * llT * O[n,nn]
                    post_prec += np.outer(self.F[nn,:], self.F[nn,:]) * llT * O[nn,n]
                    post_mean_dot_prec += O[n,nn] * zcent[n,nn] * self.F[nn,:] * self.lmbda
                    post_mean_dot_prec += O[nn,n] * zcent[nn,n] * self.F[nn,:] * self.lmbda

            # Compute the posterior mean and covariance
            post_cov  = np.linalg.inv(post_prec)
            post_mean = post_cov.dot(post_mean_dot_prec)

            # Sample from the posterior
            self.F[n,:] = np.random.multivariate_normal(post_mean, post_cov)

    def _resample_mu_0(self, As, Omegas):
        """
        Sample mu_0 from its Gaussian posterior
        """
        post_prec = 1.0 / self.sigma_mu0
        post_mean_dot_prec = self.mu_mu_0 / self.sigma_mu0

        # Add sufficient statistics from each Z
        for A,O in zip(As, Omegas):
            post_prec += O.sum()

            # Subtract off contribution from F \Lambda F
            FdotF = (self.F * self.lmbda[None,:]).dot(self.F.T)
            np.fill_diagonal(FdotF, 0)

            post_mean_dot_prec += (self.kappa(A) - O * FdotF).sum()

        # Compute the posterior mean (assuming prior mean is zero)
        post_var  = 1.0 / post_prec
        post_mean = post_var * post_mean_dot_prec

        # Sample from the posterior
        self.mu_0 =  np.random.normal(post_mean, np.sqrt(post_var))

    def _resample_lmbda(self, As, Omegas):
        """
        Sample lambda from its multivariate normal posterior
        """
        # Compute the sufficient statistics for lmbda
        post_prec = 1.0/self.sigma_lmbda * np.eye(self.D)
        post_mean_dot_prec = self.mu_lmbda / self.sigma_lmbda

        # Add up sufficient statistics for each observed graph
        for A,O in zip(As, Omegas):
            # First compute mu-mu_0
            zcent = self.kappa(A)/O - self.mu_0

            # Update the sufficient statistics
            for n1 in xrange(self.N):
                for n2 in xrange(n1+1, self.N):
                    # Update the precision
                    post_prec += (O[n2,n1] + O[n1,n2]) * \
                                 np.outer(self.F[n1,:], self.F[n1,:]) * \
                                 np.outer(self.F[n2,:], self.F[n2,:])

                    # Update the mean dot precision
                    Fn1n2 = self.F[n1,:] * self.F[n2,:]
                    post_mean_dot_prec += O[n1,n2] * zcent[n1,n2] * Fn1n2
                    post_mean_dot_prec += O[n2,n1] * zcent[n2,n1] * Fn1n2

        # Compute the posterior mean and covariance
        post_cov  = np.linalg.inv(post_prec)
        post_mean = post_cov.dot(post_mean_dot_prec)

        # Sample from the posterior
        self.lmbda = np.random.multivariate_normal(post_mean, post_cov)



class _MeanFieldLogisticEigenModel(_LogisticEigenmodelBase, MeanField):
    def __init__(self, N, D, **kwargs):
        super(_MeanFieldLogisticEigenModel, self).__init__(N, D, **kwargs)

        # Initialize mean field parameters
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

    def meanfieldupdate(self, network):
        """
        Mean field is a little different than Gibbs. We have to keep around
        variational parameters for the adjacency matrix. Thus, we assume that
        meanfieldupdate is always called with the same, single adjacency matrix.

        Update the mean field variational parameters given the expected
        adjacency matrix, E[A=1].

        :param E_A:     Expected value of the adjacency matrix
        :return:
        """
        assert isinstance(network, DirectedNetwork)
        E_A = network.E_A

        assert isinstance(E_A, np.ndarray) and E_A.shape == (self.N, self.N) \
               and np.amax(E_A) <= 1.0 and np.amin(E_A) >= 0.0

        # Compute the expectation of Omega under the variational posterior
        E_O = self._meanfieldupdate_Omega()

        # Update the latent Z's for the adjacency matrix
        self._meanfieldupdate_mu0(E_A, E_O)
        self._meanfieldupdate_F(E_A, E_O)

        if not self.lmbda_given:
            self._meanfieldupdate_lmbda(E_A, E_O)

    def _meanfieldupdate_Omega(self):
        """
        Compute the expectation of omega under the variational posterior.
        This requires us to sample activations and perform a Monte Carlo
        integration.
        """
        Zs = self.mf_sample_mus(N_samples=1000)
        E_Omega = np.ones((self.N, self.N)) / 2.0 \
                  * (np.tanh(Zs/2.0) / (Zs)).mean(axis=0)
        return E_Omega

    def _meanfieldupdate_mu0(self, E_A, E_O):
        """
        Sample mu_0 from its Gaussian posterior
        """
        post_prec = 1.0 / self.sigma_mu0
        post_mean_dot_prec = self.mu_mu_0 / self.sigma_mu0

        # Compute expected posterior parameters
        post_prec += E_O.sum()

        # Subtract off contribution from F \Lambda F
        E_F     = self.mf_mu_F
        E_lmbda = self.mf_mu_lmbda
        E_FdotF = (E_F * E_lmbda[None, :]).dot(E_F.T)
        np.fill_diagonal(E_FdotF, 0)

        post_mean_dot_prec += (self.kappa(E_A) - E_O * E_FdotF).sum()

        # Set the variational posterior parameters
        self.mf_sigma_mu0 = 1.0 / post_prec
        self.mf_mu_mu0    = self.mf_sigma_mu0 * post_mean_dot_prec

    def _meanfieldupdate_F(self, E_A, E_O):
        """
        Update the latent features.
        """
        # Update each feature given the rest
        E_mu0   = self.mf_mu_mu0
        E_lmbda = self.mf_mu_lmbda
        E_llT = self.mf_expected_llT()
        for n in xrange(self.N):
            # Compute the sufficient statistics for fn
            post_prec = 1.0/self.sigma_F * np.eye(self.D)
            post_mean_dot_prec = np.zeros(self.D)

            # First compute f * Lambda and z-mu0
            # fLambda = self.F * self.lmbda[None,:]
            zcent   = self.kappa(E_A)/E_O - E_mu0

            for nn in xrange(self.N):
                if nn == n:
                    continue

                E_ffT = self.mf_expected_ffT(nn)

                post_prec += E_ffT * E_llT * (E_O[n,nn] + E_O[nn,n])
                post_mean_dot_prec += E_O[n,nn] * zcent[n,nn] * self.F[nn,:] * E_lmbda
                post_mean_dot_prec += E_O[nn,n] * zcent[nn,n] * self.F[nn,:] * E_lmbda

            # Set the variational posterior parameters
            self.mf_Sigma_F[n,:,:] = np.linalg.inv(post_prec)
            self.mf_mu_F[n,:]      = self.mf_Sigma_F[n,:,:].dot(post_mean_dot_prec)

        assert np.all(np.isfinite(self.mf_mu_F))
        assert np.all(np.isfinite(self.mf_Sigma_F))

    def _meanfieldupdate_lmbda(self, E_A, E_O):
        """
        Mean field update for the latent feature space metric
        """
        E_F     = self.mf_mu_F
        E_mu0   = self.mf_mu_mu0

        # Compute the sufficient statistics for lmbda
        post_prec = 1.0/self.sigma_lmbda * np.eye(self.D)
        post_mean_dot_prec = self.mu_lmbda / self.sigma_lmbda

        # First compute mu-mu_0
        zcent = self.kappa(E_A)/E_O - E_mu0

        # Update the sufficient statistics
        for n1 in xrange(self.N):
            for n2 in xrange(n1+1, self.N):
                # Update the precision
                post_prec += (E_O[n2,n1] + E_O[n1,n2]) \
                             * self.mf_expected_ffT(n1) \
                             * self.mf_expected_ffT(n2)

                # Update the mean dot precision
                E_f1f2 = E_F[n1,:] * E_F[n2,:]
                post_mean_dot_prec += E_O[n1,n2] * zcent[n1,n2] * E_f1f2
                post_mean_dot_prec += E_O[n2,n1] * zcent[n2,n1] * E_f1f2

        # Compute the posterior mean and covariance
        self.mf_Sigma_lmbda = np.linalg.inv(post_prec)
        self.mf_mu_lmbda    = self.mf_Sigma_lmbda.dot(post_mean_dot_prec)

    def mf_expected_log_p(self):
        """
        Compute the expected log probability of a connection under Z
        :return:
        """
        Ps = logistic(self.mf_sample_mus())
        Ps = np.clip(Ps, 1e-16, 1-1e-16)
        return np.log(Ps).mean(0), np.log(1-Ps).mean(0)

    def mf_expected_log_notp(self):
        """
        Compute the expected log probability of a connection under Z
        :return:
        """
        Ps = logistic(self.mf_sample_mus())
        Ps = np.clip(Ps, 1e-16, 1-1e-16)
        return np.log(1-Ps).mean(0)

    def mf_expected_mu(self):
        """
        E[mu] = E[mu0 + f^T \Lambda f]
              = E[mu0] + E[f]^T E[\Lambda] E[f]
        :return:
        """
        E_mu = self.mf_mu_mu0 + (self.mf_mu_F * self.mf_mu_lmbda[None,:]).dot(self.mf_mu_F.T)

        # Override the diagonal
        np.fill_diagonal(E_mu, self.mf_mu_mu0)
        return E_mu

    def mf_expected_musq(self):
        """
        E[mu_{n1n2}^2] = E[(mu0 + f_{n1}^T \Lambda f_{n2})^2]
                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + E[(f_{n1}^T \Lambda f_{n2})^2]

                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + E[(\sum_{k} lmbda_k f_{n1, k} f_{n2, k})^2]

                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + E[\sum_{k1} \sum_{k2} f_{n1, k1} \lmbda_{k1} f_{n2,k1}
                                            x    f_{n1, k2} \lmbda_{k2} f_{n2, k2}]

                       = E[mu0^2] + 2E[mu0] E[f_{n1}]^T E[\Lambda] E[f_{n2}]
                         + sum(E[f_{n1} f_{n1}^T]  E[\lmbda \lmbda^T] E[f_{n2} f_{n2}^T])

            for n1 \neq n2
        :return:
        """
        E_musq = np.zeros((self.N, self.N))
        E_musq += self.mf_expected_mu0sq()
        E_musq += 2*self.mf_mu_mu0 * (self.mf_mu_F * self.mf_mu_lmbda[None, :]).dot(self.mf_mu_F.T)
        llT = self.mf_expected_llT()
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                E_musq[n1,n2] += np.sum(self.mf_expected_ffT(n1)
                                        * llT
                                        * self.mf_expected_ffT(n2))

        # Override the diagonal
        np.fill_diagonal(E_musq, self.mf_expected_mu0sq())

        return E_musq

    def mf_variance_mu(self):
        return self.mf_expected_musq() - self.mf_expected_mu()**2

    def mf_expected_mu0sq(self):
        return self.mf_sigma_mu0 + self.mf_mu_mu0**2

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

    def mf_sample_mus(self, N_samples=10):
        E_mu = self.mf_expected_mu()[None, :, :]
        std_mu = np.sqrt(self.mf_variance_mu())[None, :, :]
        return E_mu + std_mu * np.random.randn(N_samples, self.N, self.N)

    def expected_log_likelihood(self, network):
        """
        Compute the expected log likelihood of a graph x=A.
        :param x:
        :return:
        """
        A = network.A
        assert A.shape == (self.N, self.N) and np.all(np.bitwise_or(A==0, A==1))
        E_log_p, E_log_notp = self.mf_expected_log_p()
        return (A * E_log_p + (1-A) * E_log_notp).sum()

    def get_vlb(self):
        """
        Compute the variational lower bound. We don't know how to compute the
        entropy of the auxiliary omega's so we just compute the expected marginal
        probability of A | mu0, f, Lambda
        :return:
        """
        vlb = 0

        # E[ln p(mu0 | mu_{mu0}, sigma_{mu0})]
        vlb += ScalarGaussian(mu=self.mu_mu_0, sigmasq=self.sigma_mu0)\
               .negentropy(E_x=self.mf_mu_mu0,
                           E_xsq=self.mf_expected_mu0sq())

        # -E[ln q(mu0 | mf_mu_mu0, mf_sigma_mu0)]
        vlb -= ScalarGaussian(mu=self.mf_mu_mu0, sigmasq=self.mf_sigma_mu0).negentropy()

        # E[ln p(F | 0, sigma_{F})]
        # -E[ln q(F | mf_mu_F, mf_sigma_F)]
        for n in xrange(self.N):
            vlb += Gaussian(mu=np.zeros(self.D),
                            Sigma=self.sigma_F *np.eye(self.D))\
                .negentropy(E_x=self.mf_mu_F[n,:], E_xxT=self.mf_expected_ffT(n))

            vlb -= Gaussian(mu=self.mf_mu_F[n,:],
                            Sigma=self.mf_Sigma_F[n,:,:]).negentropy()

        # E[ln p(lmbda | mu_lmbda, sigma_{lmbda})]
        if not self.lmbda_given:
            vlb += Gaussian(mu=self.mu_lmbda,
                            Sigma=self.sigma_lmbda * np.eye(self.D))\
                .negentropy(E_x=self.mf_mu_lmbda, E_xxT=self.mf_expected_llT())

            # -E[ln q(lmbda | mf_mu_lmbda, mf_sigma_lmbda)]
            vlb -= Gaussian(mu=self.mf_mu_lmbda, Sigma=self.mf_Sigma_lmbda).negentropy()

        return vlb

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

class LogisticEigenmodel(_GibbsLogisticEigenmodel, _MeanFieldLogisticEigenModel):
    pass


# Set the default Eigenmodel
# Eigenmodel = ProbitEigenmodel
Eigenmodel = LogisticEigenmodel
