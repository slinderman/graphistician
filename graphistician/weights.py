
import numpy as np

from pybasicbayes.distributions import Gaussian, GaussianFixedMean
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

    def initialize_from_prior(self):
        pass

    def initialize_hypers(self):
        pass



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

    def initialize_from_prior(self):
        self._gaussian.resample()
        self._self_gaussian.resample()

    def initialize_hypers(self, W):
        # self.B = W.shape[2]
        mu_0 = W.mean(axis=(0,1))
        sigma_0 = np.diag(W.var(axis=(0,1)))
        self._gaussian.mu_0 = mu_0
        self._gaussian.sigma_0 = sigma_0
        self._gaussian.resample()
        # self._gaussian.nu_0 = self.B + 2

        W_self = W[np.arange(self.N), np.arange(self.N)]
        self._self_gaussian.mu_0 = W_self.mean(axis=0)
        self._self_gaussian.sigma_0 = np.diag(W_self.var(axis=0))
        self._self_gaussian.resample()
        # self._self_gaussian.nu_0 = self.B + 2

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

    def initialize_from_prior(self):
        self.m = np.random.dirichlet(self.pi)
        self.c = np.random.choice(self.C, p=self.m, size=(self.N))

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                self._gaussians[c1][c2].resample()
        if self.special_case_self_conns:
            self._self_gaussian.resample()

    def initialize_hypers(self, W):
        mu_0 = W.mean(axis=(0,1))
        sigma_0 = np.diag(W.var(axis=(0,1)))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                nu_0 = self._gaussians[c1][c2].nu_0
                self._gaussians[c1][c2].mu_0 = mu_0
                self._gaussians[c1][c2].sigma_0 = sigma_0 * (nu_0 - self.B - 1) / self.C
                self._gaussians[c1][c2].resample()

        if self.special_case_self_conns:
            W_self = W[np.arange(self.N), np.arange(self.N)]
            self._self_gaussian.mu_0 = W_self.mean(axis=0)
            self._self_gaussian.sigma_0 = np.diag(W_self.var(axis=0))
            self._self_gaussian.resample()

        # Cluster the neurons based on their rows and columns
        from sklearn.cluster import KMeans
        features = np.hstack((W[:,:,0], W[:,:,0].T))
        km = KMeans(n_clusters=self.C)
        km.fit(features)
        self.c = km.labels_.astype(np.int)

    def _get_mask(self, A, c1, c2):
            mask = ((self.c==c1)[:,None] * (self.c==c2)[None,:])
            mask &= A.astype(np.bool)
            if self.special_case_self_conns:
                mask &= True - np.eye(self.N, dtype=np.bool)

            return mask

    def log_likelihood(self, (A,W)):
        N = self.N
        assert A.shape == (N,N)
        assert W.shape == (N,N,self.B)

        ll = 0
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                mask = self._get_mask(A, c1, c2)
                ll += self._gaussians[c1][c2].log_likelihood(W[mask]).sum()

        if self.special_case_self_conns:
            mask = np.eye(self.N).astype(np.bool) & A.astype(np.bool)
            ll += self._self_gaussian.log_likelihood(W[mask]).sum()

        return ll

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

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                mask = self._get_mask(Abool, c1, c2)
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

        Abool = A.astype(np.bool)
        c_init = self.c.copy()

        def _evaluate_lkhd_slow(n1, cn1):
            ll = 0
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
                    ll += self._gaussians[cn1][cn1].log_likelihood(W[n1,n1]).sum()

                else:
                    # p(W[n1,n2] | c) and p(W[n2,n1] | c), only if there is a connection
                    if A[n1,n2]:
                        ll += self._gaussians[cn1][cn2].log_likelihood(W[n1,n2]).sum()
                    if A[n2,n1]:
                        ll += self._gaussians[cn2][cn1].log_likelihood(W[n2,n1]).sum()

            return ll

        def _evaluate_lkhd(n1, cn1):
            chat = self.c.copy()
            chat[n1] = cn1

            # Compute log lkhd for each pair of blocks
            ll = 0
            for c2 in xrange(self.C):
                # Outgoing connections
                out_mask = (chat == c2) & Abool[n1,:]
                if self.special_case_self_conns:
                    out_mask[n1] = False
                ll += self._gaussians[cn1][c2].log_likelihood(W[n1,out_mask]).sum()

                # Handle incoming connections
                # Exclude self connection since it would have been handle above
                in_mask = (chat == c2) & Abool[:,n1]
                in_mask[n1] = False
                ll += self._gaussians[c2][cn1].log_likelihood(W[in_mask,n1]).sum()

            return ll

        # Sample each assignment in order
        for n1 in xrange(self.N):
            # Compute unnormalized log probs of each connection
            lp = np.zeros(self.C)

            # Prior from m
            lp += np.log(self.m)

            # Likelihood from network
            for cn1 in xrange(self.C):
                ll = _evaluate_lkhd(n1, cn1)
                # ll_slow = _evaluate_lkhd_slow(n1, cn1)
                # assert np.allclose(ll,ll_slow)
                lp[cn1] += ll

            # Resample from lp
            self.c[n1] = sample_discrete_from_log(lp)

        # Count up the number of changes in c:
        # print "delta c: ", np.sum(1-(self.c==c_init))

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)



class LatentDistanceGaussianWeightDistribution(GaussianWeightDistribution, GibbsSampling):
    """
    l_n ~ N(0, sigma^2 I)
    W_{n', n} ~ N(A * ||l_{n'} - l_{n}||_2^2 + b, ) for n' != n
    """

    def __init__(self, N, B=1, dim=2,
                 a=1.0, b=0.0,
                 sigma=None, Sigma_0=None, nu_0=None,
                 mu_self=0.0):
        """
        Initialize SBM with parameters defined above.
        """
        super(LatentDistanceGaussianWeightDistribution, self).__init__(N)
        self.B = B
        self.dim = dim

        self.a = a
        self.b = b
        self.L = np.random.randn(N,dim)

        if Sigma_0 is None:
            Sigma_0 = np.eye(B)

        if nu_0 is None:
            nu_0 = B + 2

        self.cov = GaussianFixedMean(mu=np.zeros(B), sigma=sigma, lmbda_0=Sigma_0, nu_0=nu_0)

        # Special case self-weights (along the diagonal)
        self._self_gaussian = Gaussian(mu_0=mu_self*np.ones(B),
                                       sigma_0=Sigma_0,
                                       nu_0=nu_0,
                                       kappa_0=1.0)

    @property
    def D(self):
        return ((self.L[:, None, :] - self.L[None, :, :]) ** 2).sum(2)

    @property
    def Mu(self):
        Mu = self.a * self.D + self.b
        Mu = np.tile(Mu[:,:,None], (1,1,self.B))
        for n in xrange(self.N):
            Mu[n,n,:] = self._self_gaussian.mu

        return Mu

    @property
    def Sigma(self):
        sig = self.cov.sigma
        Sig = np.tile(sig[None,None,:,:], (self.N, self.N,1,1))

        for n in xrange(self.N):
            Sig[n,n,:,:] = self._self_gaussian.sigma

        return Sig

    def initialize_from_prior(self):
        # TODO: Initialize a and b?
        self.L = np.random.randn(self.N, self.dim)
        self.cov.resample()

    def initialize_hypers(self, A):
        pass

    def log_prior(self):
        """
        Compute the prior probability of F, mu0, and lmbda
        """
        from graphistician.internals.utils import \
            normal_inverse_wishart_log_prob, \
            inverse_wishart_log_prob
        lp = 0

        # Log prior of F under spherical Gaussian prior
        from scipy.stats import norm
        lp += norm.logpdf(self.a, 0, 1)
        lp += norm.logpdf(self.b, 0, 1)
        lp += norm.logpdf(self.L, 0, 1).sum()

        lp += inverse_wishart_log_prob(self.cov)
        lp += normal_inverse_wishart_log_prob(self._self_gaussian)

        # Log prior of mu_0 and mu_self
        return lp

    def _hmc_log_probability(self, L, a, b, A, W):
        """
        Compute the log probability as a function of L.
        This allows us to take the gradients wrt L using autograd.
        :param L:
        :param A:
        :return:
        """
        assert self.B == 1
        import autograd.numpy as anp
        # import autograd.scipy.stats.multivariate_normal as mvn
        # import ipdb; ipdb.set_trace()

        # Compute pairwise distance
        L1 = anp.reshape(L,(self.N,1,self.dim))
        L2 = anp.reshape(L,(1,self.N,self.dim))
        Mu = a * anp.sum((L1-L2)**2, axis=2) + b

        Aoff = A * (1-anp.eye(self.N))
        X = (W - Mu[:,:,None]) * Aoff[:,:,None]

        # Get the covariance and precision
        Sig = self.cov.sigma[0,0]
        Lmb = 1./Sig

        # import ipdb; ipdb.set_trace()
        lp = anp.sum(-0.5 * X**2 / Lmb)

        # lp = 0
        # for m in xrange(self.N):
        #     for n in xrange(self.N):
        #         lp += -0.5 * anp.dot(X[m,n], anp.dot(Lmb, X[m,n]))
                # lp += mvn.logpdf(W[m,n],
                #                  Mu[m,n] * anp.ones(self.B),
                #                  Sig)
                # else:
                #     lp += mvn.logpdf(W[m,n],
                #                      Mu[m,n] * anp.ones(self.B),
                #                      Sig_self)


        # Log prior of L under spherical Gaussian prior
        lp += -0.5 * anp.sum(L * L)

        # Log prior of mu0 under standardGaussian prior
        lp += -0.5 * a ** 2
        lp += -0.5 * b ** 2

        return lp

    # def rvs(self, size=[]):
    #     # Sample a network given m, c, p
    #     W = np.zeros((self.N, self.N, self.B))
    #
    #
    #     return W

    def sample_predictive_parameters(self):
        raise NotImplementedError
        # Lext = \
        #     np.vstack((self.L, np.sqrt(self.sigma) * np.random.randn(1, self.dim)))
        #
        # D = -((Lext[:,None,:] - Lext[None,:,:])**2).sum(2)
        # D += self.b
        # D += self.mu_self * np.eye(self.N+1)
        #
        # P = logistic(D)
        # Prow = P[-1,:]
        # Pcol = P[:,-1]
        # return Prow, Pcol

    def resample(self, (A,W)):
        self._resample_L(A, W)
        self._resample_A_b(A, W)
        self._resample_cov(A, W)
        self._resample_self_gaussian(A, W)

    def _resample_L(self, A, W):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc

        lp  = lambda L: self._hmc_log_probability(L, self.a, self.b, A, W)
        dlp = grad(lp)

        # import ipdb; ipdb.set_trace()

        stepsz = 0.005
        nsteps = 10
        self.L = hmc(lp, dlp, stepsz, nsteps, self.L.copy(),
                     negative_log_prob=False)

    def _resample_A_b(self, A, W):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc

        lp  = lambda (a,b): self._hmc_log_probability(self.L, a, b, A, W)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        a,b = hmc(lp, dlp, stepsz, nsteps,
                   np.array([self.a, self.b]),
                   negative_log_prob=False)
        self.a, self.b = float(a), float(b)

    def _resample_cov(self, A, W):
        # Resample covariance matrix
        Mu = self.Mu
        mask = (True-np.eye(self.N, dtype=np.bool)) & A.astype(np.bool)
        self.cov.resample(W[mask] - Mu[mask])

    def _resample_self_gaussian(self, A, W):
        # Resample self connection
        mask = np.eye(self.N, dtype=np.bool) & A.astype(np.bool)
        self._self_gaussian.resample(W[mask])

    def plot(self, A, W, ax=None, L_true=None):
        """
        If D==2, plot the embedded nodes and the connections between them

        :param L_true:  If given, rotate the inferred features to match F_true
        :return:
        """

        import matplotlib.pyplot as plt

        # Color the weights by the
        import matplotlib.cm as cm
        cmap = cm.get_cmap("RdBu")
        W_lim = abs(W[:,:,0]).max()
        W_rel = (W[:,:,0] - (-W_lim)) / (2*W_lim)

        assert self.dim==2, "Can only plot for D==2"

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, aspect="equal")

        # If true locations are given, rotate L to match L_true
        L = self.L.copy()
        if L_true is not None:
            from graphistician.internals.utils import compute_optimal_rotation
            R = compute_optimal_rotation(L, L_true)
            L = L.dot(R)

        # Scatter plot the node embeddings
        # Plot the edges between nodes
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                if A[n1,n2]:
                    ax.plot([L[n1,0], L[n2,0]],
                            [L[n1,1], L[n2,1]],
                            '-', color=cmap(W_rel[n1,n2]),
                            lw=1.0)
        ax.plot(L[:,0], L[:,1], 's', color='k', markerfacecolor='k', markeredgecolor='k')

        # Get extreme feature values
        b = np.amax(abs(L)) + L[:].std() / 2.0

        # Plot grids for origin
        ax.plot([0,0], [-b,b], ':k', lw=0.5)
        ax.plot([-b,b], [0,0], ':k', lw=0.5)

        # Set the limits
        ax.set_xlim([-b,b])
        ax.set_ylim([-b,b])

        # Labels
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        plt.show()

        return ax