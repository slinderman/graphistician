import abc
import numpy as np

class DirectedNetwork(object):
    """
    Base class for directed adjacency matrices.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def N(self):
        """
        The number of nodes
        :return:
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def A(self):
        """
        The adjacency matrix (NxN)
        :return:
        """
        raise NotImplementedError()

    # Properties required for mean field inference
    @abc.abstractproperty
    def E_A(self):
        """
        Expected adjacency matrix
        :return:
        """
        raise NotImplementedError()


class WeightedDirectedNetwork(DirectedNetwork):

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def B(self):
        """
        Dimensionality of the weights
        :return:
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def W(self):
        """
        The weight matrix (NxNxB)
        :return:
        """
        raise NotImplementedError()

    # Properties required for mean field inference
    @abc.abstractproperty
    def E_W(self):
        """
        Expected weight matrix
        :return:
        """
        raise NotImplementedError()


class GaussianWeightedDirectedNetwork(WeightedDirectedNetwork):

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def E_WWT(self):
        """
        Expected outer product for each weight matrix
        :return:
        """
        raise NotImplementedError()


class GammaWeightedDirectedNetwork(WeightedDirectedNetwork):

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def E_log_W(self):
        """
        Expected log of the weight matrix
        :return:
        """
        raise NotImplementedError()

# For convenience, define classes for fixed networks
class FixedGaussianNetwork(GaussianWeightedDirectedNetwork):
    """
    Fixed network with Gaussian weights
    """
    def __init__(self, A, W):
        N = A.shape[0]
        assert A.shape == (N,N)

        B = W.shape[2]
        assert W.shape == (N,N,B)

        self._N = N
        self._B = B
        self._A = A
        self._W = W

        self._E_A = A
        self._E_W = W
        self._E_WWT = np.zeros((N,N,B,B))
        for n1 in xrange(N):
            for n2 in xrange(N):
                self._E_WWT[n1,n2,:,:] = np.outer(W[n1,n2,:], W[n1,n2,:])

    @property
    def N(self): return self._N

    @property
    def A(self): return self._A

    @property
    def E_A(self): return self._E_A

    @property
    def W(self): return self._W

    @property
    def B(self): return self._B

    @property
    def E_W(self): return self._E_W

    @property
    def E_WWT(self): return self._E_WWT


# Now define abstractions for NetworkDistributions
class NetworkDistribution(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def P(self):
        """
        Probability of an edge
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_prior(self):
        raise NotImplementedError()

    def log_likelihood(self, network):
        """
        Compute the log likelihood of a given adjacency matrix
        :param x:
        :return:
        """
        A = network.A
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

    def log_probability(self, network):
        lp = 0
        lp += self.log_prior()
        lp += self.log_likelihood(network)
        return lp

    @abc.abstractmethod
    def resample(self, networks=[]):
        raise NotImplementedError()

    ### Mean field
    @abc.abstractmethod
    def meanfieldupdate(self, network):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_log_p(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_log_notp(self):
        raise NotImplementedError()
    #
    # def expected_log_likelihood(self, network):
    #     ell = 0
    #     ell += self.adjacency_dist.expected_log_likelihood(network)
    #     return ell
    #
    # def get_vlb(self):
    #     vlb = 0
    #     vlb  += self.adjacency_dist.get_vlb()
    #     return vlb
    #
    # def resample_from_mf(self):
    #     self.adjacency_dist.resample_from_mf()

    @abc.abstractmethod
    def expected_log_likelihood(self, network):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_vlb(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def resample_from_mf(self):
        raise NotImplementedError()


class WeightedNetworkDistribution(NetworkDistribution):
    __metaclass__ = abc.ABCMeta

    #
    # ### Mean field
    # def expected_log_likelihood(self, network):
    #     ell = super(WeightedNetworkDistribution, self).expected_log_likelihood(network)
    #     ell += self.weight_dist.expected_log_likelihood(network)
    #     return ell
    #
    # def get_vlb(self):
    #     vlb = super(WeightedNetworkDistribution, self).get_vlb()
    #     vlb  += self.adjacency_dist.get_vlb()
    #     return vlb
    #
    # def resample_from_mf(self):
    #     super(WeightedNetworkDistribution, self).resample_from_mf()
    #     self.weight_dist.resample_from_mf()


class FactorizedWeightedNetworkDistribution(WeightedNetworkDistribution):
    """
    Models where p(A,W) = p(A)p(W).
    """

    def __init__(self, N, B):
        self.N = N
        self.B = B

    @abc.abstractproperty
    def adjacency_dist(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def weight_dist(self):
        raise NotImplementedError()

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

    # def log_probability(self, x):
    #     lp = self.log_prior()
    #     lp += self.log_likelihood(x)
    #     return lp

    def rvs(self, size=[]):
        A = self.adjacency_dist.rvs(size=(self.N, self.N))
        W = self.weight_dist.rvs(size=(self.N, self.N))

        return FixedGaussianNetwork(A,W)

    # Extend the Gibbs sampling algorithm
    def resample(self, networks=[]):
        """
        Reample given a list of A's and W's
        :param data: (list of A's, list of W's)
        :return:
        """
        if isinstance(networks, WeightedDirectedNetwork):
            networks = [networks]

        self.adjacency_dist.resample(networks)
        self.weight_dist.resample(networks)

    # Extend the mean field variational inference algorithm
    def meanfieldupdate(self, network):
        """
        Reample given a list of A's and W's
        :param data: (list of A's, list of W's)
        :return:
        """
        # Mean field update the eigenmodel given As
        self.adjacency_dist.meanfieldupdate(network)

        # Mean field update the weight model
        # The "weights" of each W correspond to the probability
        # of that connection being nonzero. That is, the weights equal E_A.
        # First, convert E_W and E_WWT to be (N**2, D) and (N**2, D, D), respectively.
        weights = network.E_A
        self.weight_dist.meanfieldupdate(network, weights=weights)

    ### Mean field
    def expected_log_likelihood(self, network):
        ell = self.adjacency_dist.expected_log_likelihood(network)
        ell += self.weight_dist.expected_log_likelihood(network)
        return ell

    def get_vlb(self):
        vlb = self.adjacency_dist.get_vlb()
        vlb  += self.adjacency_dist.get_vlb()
        return vlb

    def resample_from_mf(self):
        self.adjacency_dist.resample_from_mf()
        self.weight_dist.resample_from_mf()


class GaussianWeightedNetworkDistribution(WeightedNetworkDistribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def Mu(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def Sigma(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_mu(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_mumuT(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_Sigma_inv(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_logdet_Sigma(self):
        raise NotImplementedError()


class GammaWeightedNetworkDistribution(WeightedNetworkDistribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def Kappa(self):
        """
        Shape of the gamma distributed weights
        """
        pass

    @abc.abstractproperty
    def V(self):
        """
        Inverse-Scale of the gamma-distributed weights
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_v(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mf_expected_log_v(self):
        raise NotImplementedError()
