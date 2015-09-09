### Abstractions for graphs and networks
import abc

import numpy as np
from scipy.misc import logsumexp

from pybasicbayes.abstractions import Distribution

class NetworkDistribution(Distribution):
    """
    Base class for network distributions. Distributions do not contain
    any data. Instead, they maintain a set of parameters that specify
    the distribution. With these parameters, the compute log likelihoods
    of given networks and sample random networks from the distribution.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return np.nan

    def log_probability(self, (A,W)):
        return self.log_likelihood((A,W)) + self.log_prior()
    
    def rvs(self, *args, **kwargs):
        super(NetworkDistribution, self).rvs(*args, **kwargs)

    @abc.abstractmethod
    def sample_predictive_parameters(self):
        """
        Sample a predictive set of parameters for a new row and column of the network
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictive_distribution(self):
        """
        Sample a new row and column of the network
        """
        raise NotImplementedError

    @abc.abstractmethod
    def approx_predictive_ll(self, Arow, Acol, Wrow, Wcol, M=100):
        """
        Approximate the predictive likelihood of a new row and column
        """
        raise NotImplementedError()


class AdjacencyDistribution(Distribution):
    """
    Base class for a distribution over adjacency matrices.
    Must expose a matrix of connection probabilities.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, N):
        self.N = N

    @abc.abstractproperty
    def P(self):
        """
        :return: An NxN matrix of connection probabilities.
        """
        return np.nan

    @property
    def safe_P(self):
        return np.clip(self.P, 1e-64, 1-1e-64)

    @property
    def is_deterministic(self):
        """
        Are all the entries in P either 0 or 1?
        """
        P = self.P
        return np.all(np.isclose(P,0) | np.isclose(P,1))

    @abc.abstractmethod
    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return np.nan

    def log_likelihood(self, A):
        assert A.shape == (self.N, self.N)

        P = self.safe_P
        return np.sum(A * np.log(self.P) + (1-A) * np.log(1-P))

    def log_probability(self, A):
        return self.log_likelihood(A) + self.log_prior()

    def rvs(self,size=[]):
        # TODO: Handle size
        P = self.P
        return np.random.rand(*P.shape) < P

    @abc.abstractmethod
    def sample_predictive_parameters(self):
        """
        Sample a predictive set of parameters for a new row and column of A
        :return Prow, Pcol, each an N+1 vector. By convention, the last entry
                is the new node.
        """
        raise NotImplementedError

    def sample_predictive_distribution(self):
        """
        Sample a new row and column of A
        """
        N = self.N
        Prow, Pcol = self.sample_predictive_parameters()
        # Make sure they are consistent in the (N+1)-th entry
        assert Prow[-1] == Pcol[-1]

        # Sample and make sure they are consistent in the (N+1)-th entry
        Arow = np.random.rand(N+1) < Prow
        Acol = np.random.rand(N+1) < Pcol
        Acol[-1] = Arow[-1]

        return Arow, Acol

    def approx_predictive_ll(self, Arow, Acol, M=100):
        """
        Approximate the (marginal) predictive probability by averaging over M
        samples of the predictive parameters
        """
        N = self.N
        assert Arow.shape == Acol.shape == (N+1,)
        Acol = Acol[:-1]

        # Get the predictive parameters
        lps = np.zeros(M)
        for m in xrange(M):
            Prow, Pcol = self.sample_predictive_parameters()
            Prow = np.clip(Prow, 1e-64, 1-1e-64)
            Pcol = np.clip(Pcol, 1e-64, 1-1e-64)

            # Only use the first N entries of Pcol to avoid double counting
            Pcol = Pcol[:-1]

            # Compute lp
            lps[m] += (Arow * np.log(Prow) + (1-Arow) * np.log(1-Prow)).sum()
            lps[m] += (Acol * np.log(Pcol) + (1-Acol) * np.log(1-Pcol)).sum()

        # Compute average log probability
        lp = -np.log(M) + logsumexp(lps)
        return lp


class WeightDistribution(Distribution):
    """
    Base class for a distribution over weight matrices
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, N):
        self.N = N

    @abc.abstractmethod
    def sample_predictive_parameters(self):
        """
        Sample a predictive set of parameters for a new row and column of A
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictive_parameters(self):
        """
        Sample a predictive set of parameters for a new row and column of A
        :return Prow, Pcol, each an N+1 vector. By convention, the last entry
                is the new node.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictive_distribution(self):
        """
        Sample a new row and column of A
        """
        raise NotImplementedError

    @abc.abstractmethod
    def approx_predictive_ll(self, Arow, Acol, Wrow, Wcol, M=100):
        """
        Approximate the (marginal) predictive probability by averaging over M
        samples of the predictive parameters
        """
        raise NotImplementedError

class GaussianWeightDistribution(WeightDistribution):
    """
    Base class for Guassian weight matrices.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, N, B=1):
        super(GaussianWeightDistribution, self).__init__(N)
        self.B = B

    @abc.abstractproperty
    def Mu(self):
        """
        :return: An NxNxB matrix of weight means.
        """
        return np.nan

    @abc.abstractproperty
    def Sigma(self):
        """
        :return: An NxNxBxB matrix of weight covariances.
        """
        return np.nan

    def log_likelihood(self, (A,W)):
        from scipy.stats import multivariate_normal
        N = self.N
        assert A.shape == (N,N)
        assert W.shape == (N,N,self.B)

        Mu = self.Mu
        Sig = self.Sigma

        ll = 0
        for m in xrange(N):
            for n in xrange(N):
                if A[m,n]:
                    # x = W[m,n] - Mu[m,n]
                    # ll += -0.5 * x.dot(np.linalg.solve(Sig[m,n], x))
                    ll += multivariate_normal(Mu[m,n], Sig[m,n]).pdf(W[m,n])

        return ll

    def rvs(self,size=[]):
        # TODO: Handle size
        N = self.N
        W = np.zeros((N, N, self.B))
        Mu = self.Mu
        Sig = self.Sigma

        for m in xrange(N):
            for n in xrange(N):
                W[m,n] = np.random.multivariate_normal(Mu[m,n], Sig[m,n])

        return W

    def sample_predictive_distribution(self):
        """
        Sample a new row and column of A
        """
        N = self.N
        Murow, Mucol, Lrow, Lcol = self.sample_predictive_parameters()

        # Make sure they are consistent in the (N+1)-th entry
        assert Murow[-1] == Mucol[-1]
        assert Lrow[-1] == Lcol[-1]

        # Sample and make sure they are consistent in the (N+1)-th entry
        Wrow = np.zeros((N+1, self.B))
        Wcol = np.zeros((N+1, self.B))

        for n in xrange(N+1):
            Wrow[n] = np.random.multivariate_normal(Murow[n], Lrow[n].dot(Lrow[n].T))
            if n < N:
                Wcol[n] = np.random.multivariate_normal(Mucol[n], Lcol[n].dot(Lcol[n].T))
            else:
                Wcol[n] = Wrow[n]

        return Wrow, Wcol

    def approx_predictive_ll(self, Arow, Acol, Wrow, Wcol, M=100):
        """
        Approximate the (marginal) predictive probability by averaging over M
        samples of the predictive parameters
        """
        from scipy.stats import multivariate_normal
        from numpy.core.umath_tests import inner1d
        import scipy.linalg

        N, B = self.N, self.B
        assert Arow.shape == Acol.shape == (N+1,)

        # Get the predictive parameters
        lps = np.zeros(M)
        for m in xrange(M):
            Murow, Mucol, Lrow, Lcol = self.sample_predictive_parameters()

            # for n in xrange(N+1):
            #     if Arow[n]:
            #         lps[m] += multivariate_normal(Murow[n], Sigrow[n]).pdf(Wrow[n])
            #
            #     if n < N and Acol[n]:
            #         lps[m] += multivariate_normal(Mucol[n], Sigcol[n]).pdf(Wcol[n])

            # Sigrow_chol = np.array([np.linalg.cholesky(S) for S in Sigrow])
            # Sigcol_chol = np.array([np.linalg.cholesky(S) for S in Sigcol])

            for n in xrange(N+1):
                if Arow[n]:
                    L = Lrow[n]
                    x = Wrow[n] - Murow[n]
                    xs = scipy.linalg.solve_triangular(L, x.T,lower=True)
                    lps[m] += -1./2. * inner1d(xs.T,xs.T) - B/2.*np.log(2*np.pi) \
                            - np.log(L.diagonal()).sum()

                if n < N and Acol[n]:
                    L = Lcol[n]
                    x = Wcol[n] - Mucol[n]
                    xs = scipy.linalg.solve_triangular(L, x.T,lower=True)
                    lps[m] += -1./2. * inner1d(xs.T,xs.T) - B/2.*np.log(2*np.pi) \
                            - np.log(L.diagonal()).sum()

        # Compute average log probability
        lp = -np.log(M) + logsumexp(lps)
        return lp

