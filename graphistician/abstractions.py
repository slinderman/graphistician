### Abstractions for graphs and networks
import abc

import numpy as np

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

    def log_posterior(self, networks):
        return self.log_likelihood(networks) + self.log_prior()
    
    def rvs(self, *args, **kwargs):
        super(NetworkDistribution, self).rvs(*args, **kwargs)


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

    def log_likelihood(self, A):
        assert A.shape == (self.N, self.N)

        P = self.safe_P
        return np.sum(A * np.log(self.P) + (1-A) * np.log(1-P))

    def rvs(self,size=[]):
        # TODO: Handle size
        P = self.P
        return np.random.rand(*P.shape) < P


class WeightDistribution(Distribution):
    """
    Base class for a distribution over weight matrices
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, N):
        self.N = N


class NullWeightDistribution(WeightDistribution):
    """
    Dummy class for unweighted networks
    """
    def log_likelihood(self, (A,W)):
        return 0

    def rvs(self,size=[]):
        return None


class GaussianWeightDistribution(WeightDistribution):
    """
    Base class for Guassian weight matrices.
    """
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
        N = self.N
        assert A.shape == (N,N)
        assert W.shape == (N,N,self.B)

        Mu = self.Mu
        Sig = self.Sigma

        ll = 0
        for m in xrange(N):
            for n in xrange(N):
                if A[m,n]:
                    x = W[m,n] - Mu[m,n]
                    ll += -0.5 * x.dot(np.linalg.solve(Sig[m,n], x))

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


class FactorizedNetworkDistribution(NetworkDistribution):
    def __init__(self, N,
                 adjacency_class, adjacency_kwargs,
                 weight_class, weight_kwargs):

        super(FactorizedNetworkDistribution, self).__init__()
        assert issubclass(adjacency_class, AdjacencyDistribution)
        self.adjacency = adjacency_class(N, **adjacency_kwargs)
        assert issubclass(weight_class, WeightDistribution)
        self.weights = weight_class(N, **weight_kwargs)

    def log_likelihood(self, networks):
        if isinstance(networks, list):
            lls = []
            for A,W in networks:
                ll = self.adjacency.log_likelihood(A)
                ll += self.weights.log_likelihood((A,W))
                lls.append(ll)

        else:
            A,W = networks
            lls = self.adjacency.log_likelihood(A)
            lls += self.weights.log_likelihood((A,W))

        return lls

    def rvs(self, *args, **kwargs):
        A = self.adjacency.rvs()
        W = self.weights.rvs()
        return A,W