from pybasicbayes.abstractions import GibbsSampling

from abstractions import NetworkDistribution, AdjacencyDistribution, WeightDistribution
import adjacency
import weights

class FactorizedNetworkDistribution(NetworkDistribution, GibbsSampling):
    def __init__(self, N,
                 adjacency_class, adjacency_kwargs,
                 weight_class, weight_kwargs):

        super(FactorizedNetworkDistribution, self).__init__()
        assert issubclass(adjacency_class, AdjacencyDistribution)
        self.adjacency = adjacency_class(N, **adjacency_kwargs)
        assert issubclass(weight_class, WeightDistribution)
        self.weights = weight_class(N, **weight_kwargs)

    def log_likelihood(self, (A,W)):
        lls = self.adjacency.log_likelihood(A)
        lls += self.weights.log_likelihood((A,W))
        return lls

    def log_prior(self):
        lp = 0
        lp += self.adjacency.log_prior()
        lp += self.weights.log_prior()
        return lp

    def rvs(self, *args, **kwargs):
        A = self.adjacency.rvs()
        W = self.weights.rvs()
        return A,W

    def resample(self, (A,W)):
        self.adjacency.resample(A)
        self.weights.resample((A,W))


# Unweighted network Models
class UnweightedBernoulliModel(FactorizedNetworkDistribution):
    def __init__(self, N, p):
        super(UnweightedBernoulliModel, self).\
            __init__(N,
                     adjacency.BernoulliAdjacencyDistribution, {"p": p},
                     weights.NullWeightDistribution, {})

class UnweightedBetaBernoulliModel(FactorizedNetworkDistribution):
    def __init__(self, N, **kwargs):
        super(UnweightedBetaBernoulliModel, self).\
            __init__(N,
                     adjacency.BetaBernoulliAdjacencyDistribution, kwargs,
                     weights.NullWeightDistribution, {})

class UnweightedLatentDistanceModel(FactorizedNetworkDistribution):
    def __init__(self, N, **kwargs):
        super(UnweightedLatentDistanceModel, self).\
            __init__(N,
                     adjacency.LatentDistanceAdjacencyDistribution, kwargs,
                     weights.NullWeightDistribution, {})

    def plot(self, A, **kwargs):
        self.adjacency.plot(A, **kwargs)


