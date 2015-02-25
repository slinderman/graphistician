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

    @abc.abstractmethod
    def log_likelihood(self, networks=[]):
        raise NotImplementedError()

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

    @abc.abstractmethod
    def get_vlb(self):
        pass


class GaussianWeightedNetworkDistribution(NetworkDistribution):
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
