
import numpy as np

from pybasicbayes.distributions import Gaussian
from pybasicbayes.abstractions import GibbsSampling

from abstractions import WeightDistribution, GaussianWeightDistribution


class FixedGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    def __init__(self, N, B, mu, sigma):
        super(FixedGaussianWeightDistribution, self).__init__(N)
        self.B = B

        assert mu.shape == (B,)
        self.mu = mu

        assert sigma.shape == (B,B)
        self.sigma = sigma

        self._gaussian = Gaussian(mu, sigma)

    @property
    def Mu(self):
        raise NotImplementedError()

    @property
    def Sigma(self):
        raise NotImplementedError()

    def log_prior(self):
        return 0

    def resample(self, (A,W)):
        pass


class NIWGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    """
    Gaussian weight distribution with a normal inverse-Wishart prior.
    """
    def __init__(self, N, B, mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None):
        super(NIWGaussianWeightDistribution, self).__init__(N)
        self.B = B


        if mu_0 is None:
            mu_0 = np.zeros(B)

        if Sigma_0 is None:
            Sigma_0 = np.eye(B)

        if nu_0 is None:
            nu_0 = B + 2

        if kappa_0 is None:
            kappa_0 = 1.0

        self._gaussian = Gaussian(mu_0=mu_0, sigma_0=Sigma_0,
                                  nu_0=nu_0, kappa_0=kappa_0)

    @property
    def Mu(self):
        raise NotImplementedError()

    @property
    def Sigma(self):
        raise NotImplementedError()

    def log_prior(self):
        # TODO: Compute log prior of Normal-Inverse Wishart
        return 0

    def resample(self, (A,W)):
        # Resample the Normal-inverse Wishart prior over mu and W
        # given W for which A=1
        self._gaussian.resample(W[A==1])


class LowRankGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    """
    Low rank weight matrix (i.e. BPMF from Minh and Salakhutidnov)
    """
    pass

class SBMGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    """
    A stochastic block model is a clustered network model with
    C:          Number of blocks
    m[c]:       Probability that a node belongs block c
    mu[c,c']:   Mean weight from node in block c to node in block c'
    Sig[c,c']:  Cov of weight from node in block c to node in block c'

    It has hyperparameters:
    pi:         Parameter of Dirichlet prior over m
    mu0, nu0, kappa0, Sigma0: Parameters of NIW prior over (mu,Sig)
    """

    # Override this in base classes!
    _weight_class = None
    _default_weight_hypers = {}

    def __init__(self, N, B,
                 C=1,
                 pi=1.0,
                 mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None):
        """
        Initialize SBM with parameters defined above.
        """
        super(SBMGaussianWeightDistribution, self).__init__(N)
        self.B = B

        assert isinstance(C, int) and C >= 1, "C must be a positive integer number of blocks"
        self.C = C

        if isinstance(pi, (int, float)):
            self.pi = pi * np.ones(C)
        else:
            assert isinstance(pi, np.ndarray) and pi.shape == (C,), "pi must be a sclar or a C-vector"
            self.pi = pi

        self.m = np.random.dirichlet(self.pi)
        self.c = np.random.choice(self.C, p=self.m, size=(self.N))

        self._gaussians = [[Gaussian(mu_0=mu_0, nu_0=nu_0,
                                     kappa_0=kappa_0, sigma_0=Sigma_0)
                            for _ in xrange(C)]
                           for _ in xrange(C)]

    @property
    def _Mu(self):
        # TODO: Double check this
        return np.array([[self._gaussians[c1][c2].mu
                          for c2 in xrange(self.C)]
                         for c1 in xrange(self.C)])

    @property
    def _Sigma(self):
        # TODO: Double check this
        return np.array([[self._gaussians[c1][c2].sigma
                          for c2 in xrange(self.C)]
                         for c1 in xrange(self.C)])

    @property
    def Mu(self):
        """
        Get the NxNxB matrix of weight means
        :return:
        """
        _Mu = self._Mu
        Mu = _Mu[np.ix_(self.c, self.c)]
        # if not self.allow_self_connections:
        #     np.fill_diagonal(P, 0.0)
        return Mu

    @property
    def Sigma(self):
        """
        Get the NxNxBxB matrix of weight covariances
        :return:
        """
        _Sigma = self._Sigma
        Sigma = _Sigma[np.ix_(self.c, self.c)]
        # if not self.allow_self_connections:
        #     np.fill_diagonal(P, 0.0)
        return Sigma

    def log_prior(self):
        """
        Compute the log likelihood of a set of SBM parameters

        :param x:    (m,p,v) tuple
        :return:
        """
        lp = 0
        lp += Dirichlet(self.pi).log_probability(self.m)
        # lp += Beta(self.tau1 * np.ones((self.C, self.C)),
        #            self.tau0 * np.ones((self.C, self.C))).\
        #     log_probability(self.p).sum()
        lp += (np.log(self.m)[self.c]).sum()
        return lp


    def rvs(self, size=[]):
        # Sample a network given m, c, p
        A = np.zeros((self.N, self.N))

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                blk = (self.c==c1)[:,None] * (self.c==c2)[None,:]
                A[blk] = np.random.rand(blk.sum()) < self.p[c1,c2]

        return A

    ###
    ### Implement Gibbs sampling for SBM
    ###
    def resample(self, (A,W)):
        self.resample_mu_and_Sig(A,W)
        self.resample_c(A,W)
        self.resample_m()

    def resample_mu_and_Sig(self, A, W):
        """
        Resample p given observations of the weights
        """
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                mask = ((self.c==c1)[:,None] * (self.c==c2)[None,:]) & A
                self._gaussians[c1][c2].resample(W[mask])

    def resample_c(self, A, W):
        """
        Resample block assignments given the weighted adjacency matrix
        """
        if self.C == 1:
            return

        # Sample each assignment in order
        for n1 in xrange(self.N):
            # Compute unnormalized log probs of each connection
            lp = np.zeros(self.C)

            # Prior from m
            lp += np.log(self.m)

            # Likelihood from network
            for cn1 in xrange(self.C):

                # Compute probability for each incoming and outgoing
                for n2 in xrange(self.N):
                    cn2 = self.c[n2]

                    if A[n1,n2]:
                        if n2 != n1:
                            # p(W[n1,n2] | c)
                            lp[cn1] += self._gaussians[cn1][cn2].log_likelihood(W[n1,n2]).sum()

                            # p(A[n2,n1] | c)
                            lp[cn1] += self._gaussians[cn2][cn1].log_likelihood(W[n2,n1]).sum()

                        else:
                            # Self connection
                            lp[cn1] += self._gaussians[cn1][cn1].log_likelihood(W[n1,n1]).sum()

            # Resample from lp
            self.c[n1] = sample_discrete_from_log(lp)

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)
