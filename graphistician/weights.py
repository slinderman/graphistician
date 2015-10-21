
import numpy as np

from pybasicbayes.distributions import Gaussian
from pybasicbayes.abstractions import GibbsSampling

from abstractions import GaussianWeightDistribution

class FixedGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    def __init__(self, N, B, mu, sigma, mu_self=None, sigma_self=None):
        super(FixedGaussianWeightDistribution, self).__init__(N)
        self.B = B

        assert mu.shape == (B,)
        self.mu = mu

        assert sigma.shape == (B,B)
        self.sigma = sigma

        self._gaussian = Gaussian(mu, sigma)

        if mu_self is not None and sigma_self is not None:
            self._self_gaussian = Gaussian(mu_self, sigma_self)
        else:
            self._self_gaussian = self._gaussian


    @property
    def Mu(self):
        mu = self._gaussian.mu
        Mu = np.tile(mu[None,None,:], (self.N, self.N,1))

        for n in xrange(self.N):
            Mu[n,n,:] = self._self_gaussian.mu

        return Mu

    @property
    def Sigma(self):
        sig = self._gaussian.sigma
        Sig = np.tile(sig[None,None,:,:], (self.N, self.N,1,1))

        for n in xrange(self.N):
            Sig[n,n,:,:] = self._self_gaussian.sigma

        return Sig


    def log_prior(self):
        return 0

    def sample_predictive_parameters(self):
        Murow = Mucol = np.tile(self._gaussian.mu[None,:], (self.N+1,1))
        Lrow = Lcol = np.tile(self._gaussian.sigma_chol[None,:,:], (self.N+1,1,1))
        return Murow, Mucol, Lrow, Lcol

    def resample(self, (A,W)):
        pass


class NIWGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    """
    Gaussian weight distribution with a normal inverse-Wishart prior.
    """
    # TODO: Specify the self weight parameters in the constructor
    def __init__(self, N, B=1, mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None):
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

        # Special case self-weights (along the diagonal)
        self._self_gaussian = Gaussian(mu_0=mu_0, sigma_0=Sigma_0,
                                       nu_0=nu_0, kappa_0=kappa_0)

    @property
    def Mu(self):
        mu = self._gaussian.mu
        Mu = np.tile(mu[None,None,:], (self.N, self.N,1))

        for n in xrange(self.N):
            Mu[n,n,:] = self._self_gaussian.mu

        return Mu

    @property
    def Sigma(self):
        sig = self._gaussian.sigma
        Sig = np.tile(sig[None,None,:,:], (self.N, self.N,1,1))

        for n in xrange(self.N):
            Sig[n,n,:,:] = self._self_gaussian.sigma

        return Sig

    def log_prior(self):
        from graphistician.internals.utils import normal_inverse_wishart_log_prob
        lp = 0
        lp += normal_inverse_wishart_log_prob(self._gaussian)
        lp += normal_inverse_wishart_log_prob(self._self_gaussian)

        return lp

    def sample_predictive_parameters(self):
        Murow = Mucol = np.tile(self._gaussian.mu[None,:], (self.N+1,1))
        Lrow = Lcol = np.tile(self._gaussian.sigma_chol[None,:,:], (self.N+1,1,1))

        Murow[-1,:] = self._self_gaussian.mu
        Mucol[-1,:] = self._self_gaussian.mu
        Lrow[-1,:,:] = self._self_gaussian.sigma_chol
        Lcol[-1,:,:] = self._self_gaussian.sigma_chol
        return Murow, Mucol, Lrow, Lcol

    def resample(self, (A,W)):
        # Resample the Normal-inverse Wishart prior over mu and W
        # given W for which A=1
        A_offdiag = A.copy()
        np.fill_diagonal(A_offdiag, 0)

        A_ondiag = A * np.eye(self.N)
        self._gaussian.resample(W[A_offdiag==1])
        self._self_gaussian.resample(W[A_ondiag==1])


class LowRankGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    """
    Low rank weight matrix (i.e. BPMF from Mnih and Salakhutidnov)
    """
    def __init__(self, N, dim):
        raise NotImplementedError

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

    # TODO: Specify the self weight parameters in the constructor
    def __init__(self, N, B=1,
                 C=2, pi=10.0,
                 mu_0=None, Sigma_0=None, nu_0=None, kappa_0=None,
                 special_case_self_conns=True):
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

        if mu_0 is None:
            mu_0 = np.zeros(B)

        if Sigma_0 is None:
            Sigma_0 = np.eye(B)

        if nu_0 is None:
            nu_0 = B + 2

        if kappa_0 is None:
            kappa_0 = 1.0

        self._gaussians = [[Gaussian(mu_0=mu_0, nu_0=nu_0,
                                     kappa_0=kappa_0, sigma_0=Sigma_0)
                            for _ in xrange(C)]
                           for _ in xrange(C)]

        # Special case self-weights (along the diagonal)
        self.special_case_self_conns = special_case_self_conns
        if special_case_self_conns:
            self._self_gaussian = Gaussian(mu_0=mu_0, sigma_0=Sigma_0,
                                           nu_0=nu_0, kappa_0=kappa_0)

    @property
    def _Mu(self):
        return np.array([[self._gaussians[c1][c2].mu
                          for c2 in xrange(self.C)]
                         for c1 in xrange(self.C)])

    @property
    def _Sigma(self):
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

        if self.special_case_self_conns:
            for n in xrange(self.N):
                Mu[n,n] = self._self_gaussian.mu

        return Mu

    @property
    def Sigma(self):
        """
        Get the NxNxBxB matrix of weight covariances
        :return:
        """
        _Sigma = self._Sigma
        Sigma = _Sigma[np.ix_(self.c, self.c)]

        if self.special_case_self_conns:
            for n in xrange(self.N):
                Sigma[n,n] = self._self_gaussian.sigma

        return Sigma

    def log_prior(self):
        """
        Compute the log likelihood of a set of SBM parameters

        :param x:    (m,p,v) tuple
        :return:
        """
        from scipy.stats import dirichlet
        from graphistician.internals.utils import normal_inverse_wishart_log_prob
        lp = 0

        # Get the log probability of the block probabilities
        lp += dirichlet(self.pi).logpdf(self.m)

        # Get the prior probability of the Gaussian parameters under NIW prior
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                lp += normal_inverse_wishart_log_prob(self._gaussians[c1][c2])

        if self.special_case_self_conns:
            lp += normal_inverse_wishart_log_prob(self._self_gaussian)

        # Get the probability of the block assignments
        lp += (np.log(self.m)[self.c]).sum()
        return lp


    def rvs(self, size=[]):
        # Sample a network given m, c, p
        W = np.zeros((self.N, self.N, self.B))

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                blk = (self.c==c1)[:,None] * (self.c==c2)[None,:]
                W[blk] = self._gaussians[c1][c2].rvs(size=blk.shape)

        if self.special_case_self_conns:
            for n in xrange(self.N):
                W[n,n] = self._self_gaussian.rvs()

        return W

    def sample_predictive_parameters(self):
        # Sample a new cluster assignment
        c2 = np.random.choice(self.C, p=self.m)
        cext = np.concatenate((self.c, [c2]))

        Murow = np.array([self._gaussians[c2][c1].mu for c1 in cext])
        Lrow  = np.array([self._gaussians[c2][c1].sigma_chol for c1 in cext])
        Mucol = np.array([self._gaussians[c1][c2].mu for c1 in cext])
        Lcol = np.array([self._gaussians[c1][c2].sigma_chol for c1 in cext])

        if self.special_case_self_conns:
            Murow[-1] = Mucol[-1] = self._self_gaussian.mu
            Lrow[-1] = Lcol[-1] = self._self_gaussian.sigma_chol

        return Murow, Mucol, Lrow, Lcol

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
        Abool = A.astype(np.bool)
        def _get_mask(c1, c2):
            mask = ((self.c==c1)[:,None] * (self.c==c2)[None,:])
            mask &= Abool
            if self.special_case_self_conns:
                mask &= True - np.eye(self.N, dtype=np.bool)

            return mask

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                mask = _get_mask(c1, c2)
                self._gaussians[c1][c2].resample(W[mask])

        # Resample self connection
        if self.special_case_self_conns:
            mask = np.eye(self.N, dtype=np.bool) & Abool
            self._self_gaussian.resample(W[mask])

    def resample_c(self, A, W):
        """
        Resample block assignments given the weighted adjacency matrix
        """
        from pybasicbayes.util.stats import sample_discrete_from_log

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

                    # If we are special casing the self connections then
                    # we can just continue if n1==n2 since its weight has
                    # no bearing on the cluster assignment
                    if n2 == n1:
                        # Self connection
                        if self.special_case_self_conns:
                            continue
                        lp[cn1] += self._gaussians[cn1][cn1].log_likelihood(W[n1,n1]).sum()

                    else:
                        # p(W[n1,n2] | c) and p(W[n2,n1] | c), only if there is a connection
                        if A[n1,n2]:
                            lp[cn1] += self._gaussians[cn1][cn2].log_likelihood(W[n1,n2]).sum()
                        if A[n2,n1]:
                            lp[cn1] += self._gaussians[cn2][cn1].log_likelihood(W[n2,n1]).sum()

            # Resample from lp
            self.c[n1] = sample_discrete_from_log(lp)

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)
