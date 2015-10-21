"""
Super simple adjacency models. We either have a Bernoulli model
with fixed probability or a beta-Bernoulli model.
"""
import numpy as np
from scipy.stats import beta

from abstractions import AdjacencyDistribution
from internals.utils import logistic

from pybasicbayes.abstractions import GibbsSampling

class FixedAdjacencyDistribution(AdjacencyDistribution, GibbsSampling):
    """
    Simple class for a fixed adjacency matrix
    """
    def __init__(self, N, A):
        super(FixedAdjacencyDistribution, self).__init__(N)
        assert A.shape == (N,N)
        assert np.all((A==1) | (A==0))
        self._P = A

    @property
    def P(self):
        return self._P

    def log_prior(self):
        return 0

    def resample(self,data=[]):
        pass

    def sample_predictive_parameters(self):
        raise Exception("Cannot sample parameters for FixedAdjacencyDistribution")


class CompleteAdjacencyDistribution(FixedAdjacencyDistribution):
    def __init__(self, N):
        super(CompleteAdjacencyDistribution, self).__init__(N, np.ones((N,N)))

    def sample_predictive_parameters(self):
        return np.ones(self.N+1), np.ones(self.N+1)


class EmptyAdjacencyDistribution(FixedAdjacencyDistribution):
    def __init__(self, N):
        super(EmptyAdjacencyDistribution, self).__init__(N, np.zeros((N,N)))

    def sample_predictive_parameters(self):
        return np.zeros(self.N+1), np.zeros(self.N+1)


class BernoulliAdjacencyDistribution(AdjacencyDistribution, GibbsSampling):
    """
    Bernoulli edge model with fixed probability
    """
    def __init__(self, N, p, p_self=None):
        super(BernoulliAdjacencyDistribution, self).__init__(N)

        assert p > 0 and p < 1
        self.p = p
        self.p_self = p_self if p_self else p
        self._P = p * np.ones((N,N))
        np.fill_diagonal(self._P, self.p_self)


    @property
    def P(self):
        return self._P

    def log_prior(self):
        return 0

    def resample(self, data=[]):
        pass

    def sample_predictive_parameters(self):
        p_row, p_col = self.p * np.ones(self.N+1), self.p * np.ones(self.N+1)
        p_row[-1] = self.p_self
        p_col[-1] = self.p_self

        return p_row, p_col


class BetaBernoulliAdjacencyDistribution(AdjacencyDistribution, GibbsSampling):
    def __init__(self, N, tau1=1.0, tau0=1.0, tau1_self=None, tau0_self=None):
        super(BetaBernoulliAdjacencyDistribution, self).__init__(N)
        assert tau1 > 0 and tau0 > 0
        self.tau1 = tau1
        self.tau0 = tau0

        self.p = np.random.beta(tau1, tau0)

        if tau1_self is not None and tau0_self is not None:
            self.self_connection = True
            self.tau1_self = tau1_self
            self.tau0_self = tau0_self
            self.p_self = np.random.beta(tau1_self, tau0_self)
        else:
            self.self_connection = False


    @property
    def P(self):
        P = self.p * np.ones((self.N, self.N))
        if self.self_connection:
            np.fill_diagonal(P, self.p_self)
        return P

    def rvs(self, size=[]):
        return np.random.rand(self.N, self.N) < self.P

    def log_prior(self):
        lp = beta(self.tau1, self.tau0).logpdf(self.p)
        if self.self_connection:
            lp += beta(self.tau1_self, self.tau0_self).logpdf(self.p_self)

        return lp

    def resample(self, A):
        """
        Resample p given observations of the weights
        """

        def _posterior_params(_A, mask, t1, t0):
            n_conns = _A[mask].sum()
            n_noconns = mask.sum() - n_conns

            t1p = t1 + n_conns
            t0p = t0 + n_noconns

            return t1p, t0p

        if self.self_connection:
            # First update off diagonal probability
            mask = (1 - np.eye(self.N)).astype(np.bool)
            t1p, t0p = _posterior_params(A, mask, self.tau1, self.tau0)
            self.p = np.random.beta(t1p, t0p)

            # Then update self probability
            mask = np.eye(self.N).astype(np.bool)
            t1p, t0p = _posterior_params(A, mask, self.tau1, self.tau0)
            self.p_self = np.random.beta(t1p, t0p)

        else:
            mask = np.ones((self.N, self.N), dtype=np.bool)
            t1p, t0p = _posterior_params(A, mask, self.tau1, self.tau0)
            self.p = np.random.beta(t1p, t0p)

    def sample_predictive_parameters(self):
        p_row, p_col = self.p * np.ones(self.N+1), self.p * np.ones(self.N+1)
        if self.self_connection:
            p_row[-1] = p_col[-1] = self.p_self

        return p_row, p_col


class LatentDistanceAdjacencyDistribution(AdjacencyDistribution, GibbsSampling):
    """
    l_n ~ N(0, sigma^2 I)
    A_{n', n} ~ Bern(\sigma(-||l_{n'} - l_{n}||_2^2))
    """
    def __init__(self, N, dim=2, sigma=1.0, mu0=0.0, mu_self=0.0):
        self.N = N
        self.dim = dim
        self.sigma = sigma
        self.mu_0 = mu0
        self.mu_self = mu_self
        self.L = np.sqrt(self.sigma) * np.random.randn(N,dim)

    @property
    def D(self):
        Mu = -((self.L[:,None,:] - self.L[None,:,:])**2).sum(2)
        Mu += self.mu_0
        Mu += self.mu_self * np.eye(self.N)

        return Mu

    @property
    def P(self):
        P = logistic(self.D)
        return P

    def log_prior(self):
        """
        Compute the prior probability of F, mu0, and lmbda
        """
        lp  = 0

        # Log prior of F under spherical Gaussian prior
        from scipy.stats import norm
        lp += norm.logpdf(self.L, 0, np.sqrt(self.sigma)).sum()

        # Log prior of mu_0 and mu_self
        lp += norm.logpdf(self.mu_0, 0, 1)
        lp += norm.logpdf(self.mu_self, 0, 1)
        return lp

    def _hmc_log_probability(self, L, mu_0, mu_self, A):
        """
        Compute the log probability as a function of L.
        This allows us to take the gradients wrt L using autograd.
        :param L:
        :param A:
        :return:
        """
        import autograd.numpy as anp
        # Compute pairwise distance
        L1 = anp.reshape(L,(self.N,1,self.dim))
        L2 = anp.reshape(L,(1,self.N,self.dim))
        D = - anp.sum((L1-L2)**2, axis=2)

        # Compute the logit probability
        logit_P = D + mu_0 + mu_self * np.eye(self.N)

        # Take the logistic of the negative distance
        P = 1.0 / (1+anp.exp(-logit_P))

        # Compute the log likelihood
        ll = anp.sum(A * anp.log(P) + (1-A) * anp.log(1-P))

        # Log prior of L under spherical Gaussian prior
        lp = -0.5 * anp.sum(L * L / self.sigma)

        # Log prior of mu0 under standardGaussian prior
        lp += -0.5 * mu_0**2

        lp += -0.5 * mu_self**2

        return ll + lp


    def rvs(self, size=[]):
        """
        Sample a new NxN network with the current distribution parameters

        :param size:
        :return:
        """
        # TODO: Sample the specified number of graphs
        P = self.P
        A = np.random.rand(self.N, self.N) < P

        return A

    def sample_predictive_parameters(self):
        Lext = \
            np.vstack((self.L, np.sqrt(self.sigma) * np.random.randn(1, self.dim)))

        D = -((Lext[:,None,:] - Lext[None,:,:])**2).sum(2)
        D += self.mu_0
        D += self.mu_self * np.eye(self.N+1)

        P = logistic(D)
        Prow = P[-1,:]
        Pcol = P[:,-1]

        return Prow, Pcol

    def compute_optimal_rotation(self, L_true):
        """
        Find a rotation matrix R such that F_inf * R ~= F_true
        :return:
        """
        from scipy.linalg import orthogonal_procrustes
        R = orthogonal_procrustes(self.L, L_true)[0]
        return R

    def plot(self, A, ax=None, color='k', L_true=None, lmbda_true=None):
        """
        If D==2, plot the embedded nodes and the connections between them

        :param L_true:  If given, rotate the inferred features to match F_true
        :return:
        """

        import matplotlib.pyplot as plt

        assert self.dim==2, "Can only plot for D==2"

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, aspect="equal")

        # If true locations are given, rotate L to match L_true
        L = self.L
        if L_true is not None:
            R = self.compute_optimal_rotation(L_true)
            L = L.dot(R)

        # Scatter plot the node embeddings
        ax.plot(L[:,0], L[:,1], 's', color=color, markerfacecolor=color, markeredgecolor=color)
        # Plot the edges between nodes
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                if A[n1,n2]:
                    ax.plot([L[n1,0], L[n2,0]],
                            [L[n1,1], L[n2,1]],
                            '-', color=color, lw=1.0)

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

    def resample(self, A):
        """
        Resample the parameters of the distribution given the observed graphs.
        :param data:
        :return:
        """
        # Sample the latent positions
        self._resample_L(A)

        # Resample the offsets
        self._resample_mu_0(A)
        self._resample_mu_self(A)

    def _resample_L(self, A):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc

        lp  = lambda L: self._hmc_log_probability(L, self.mu_0, self.mu_self, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        self.L = hmc(lp, dlp, stepsz, nsteps, self.L.copy(), negative_log_prob=False)

    def _resample_mu_0(self, A):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc


        lp  = lambda mu_0: self._hmc_log_probability(self.L, mu_0, self.mu_self, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        mu_0 = hmc(lp, dlp, stepsz, nsteps, np.array(self.mu_0), negative_log_prob=False)
        self.mu_0 = float(mu_0)

    def _resample_mu_self(self, A):
        """
        Resample the self connection offset
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc


        lp  = lambda mu_self: self._hmc_log_probability(self.L, self.mu_0, mu_self, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        mu_self = hmc(lp, dlp, stepsz, nsteps, np.array(self.mu_self), negative_log_prob=False)
        self.mu_self = float(mu_self)


class SBMAdjacencyDistribution(AdjacencyDistribution, GibbsSampling):
    """
    A stochastic block model is a clustered network model with
    K:          Number of nodes in the network
    C:          Number of blocks
    m[c]:       Probability that a node belongs block c
    p[c,c']:    Probability of connection from node in block c to node in block c'

    It is parameterized by:
    pi:         Parameter of Dirichlet prior over m
    tau0, tau1: Parameters of beta prior over p
    """

    # Override this in base classes!
    _weight_class = None
    _default_weight_hypers = {}

    def __init__(self, N,
                 C=2,
                 c=None, m=None, pi=1.0,
                 p=None, tau0=1.0, tau1=1.0,
                 allow_self_connections=True,
                 special_case_self_conns=True):
        """
        Initialize SBM with parameters defined above.
        """
        super(SBMAdjacencyDistribution, self).__init__(N)

        assert isinstance(C, int) and C >= 1, "C must be a positive integer number of blocks"
        self.C = C

        if isinstance(pi, (int, float)):
            self.pi = pi * np.ones(C)
        else:
            assert isinstance(pi, np.ndarray) and pi.shape == (C,), "pi must be a sclar or a C-vector"
            self.pi = pi

        self.tau0    = tau0
        self.tau1    = tau1
        self.allow_self_connections = allow_self_connections

        if m is not None:
            assert isinstance(m, np.ndarray) and m.shape == (C,) \
                   and np.allclose(m.sum(), 1.0) and np.amin(m) >= 0.0, \
                "m must be a length C probability vector"
            self.m = m
        else:
            self.m = np.random.dirichlet(self.pi)


        if c is not None:
            assert isinstance(c, np.ndarray) and c.shape == (self.N,) and c.dtype == np.int \
                   and np.amin(c) >= 0 and np.amax(c) <= self.C-1, \
                "c must be a length K-vector of block assignments"
            self.c = c.copy()
        else:
            self.c = np.random.choice(self.C, p=self.m, size=(self.N))

        if p is not None:
            if np.isscalar(p):
                assert p >= 0 and p <= 1, "p must be a probability"
                self.p = p * np.ones((C,C))

            else:
                assert isinstance(p, np.ndarray) and p.shape == (C,C) \
                       and np.amin(p) >= 0 and np.amax(p) <= 1.0, \
                    "p must be a CxC matrix of probabilities"
                self.p = p
        else:
            self.p = np.random.beta(self.tau1, self.tau0, size=(self.C, self.C))

        # Special case self-weights (along the diagonal)
        self.special_case_self_conns = special_case_self_conns
        if special_case_self_conns:
            self.p_self = np.random.beta(self.tau1, self.tau0)

    @property
    def P(self):
        """
        Get the KxK matrix of probabilities
        :return:
        """
        P = self.p[np.ix_(self.c, self.c)]
        if self.special_case_self_conns:
            np.fill_diagonal(P, self.p_self)
        if not self.allow_self_connections:
            np.fill_diagonal(P, 0.0)

        return P

    def log_prior(self):
        """
        Compute the log likelihood of a set of SBM parameters

        :param x:    (m,p,v) tuple
        :return:
        """
        from scipy.stats import dirichlet, beta
        lp = 0
        lp += dirichlet(self.pi).logpdf(self.m)

        lp += beta(self.tau1 * np.ones((self.C, self.C)),
                   self.tau0 * np.ones((self.C, self.C))).logpdf(self.p).sum()
        if self.special_case_self_conns:
            lp += beta(self.tau1, self.tau0).logpdf(self.p_self)

        lp += (np.log(self.m)[self.c]).sum()
        return lp


    def rvs(self, size=[]):
        A = np.random.rand(self.N, self.N) < self.P
        return A

    def sample_predictive_parameters(self):
        # Sample a new cluster assignment
        cext = np.concatenate((self.c, [np.random.choice(self.C, p=self.m)]))

        P = self.p[np.ix_(cext, cext)]
        if not self.allow_self_connections:
            np.fill_diagonal(P, 0.0)

        Prow = P[-1,:]
        Pcol = P[:,-1]

        return Prow, Pcol

    def invariant_sbm_order(self):
        """
        Return an (almost) invariant ordering of the block labels
        """
        # Get the counts for each cluster
        blocksz = np.bincount(self.c, minlength=self.C)

        # Sort by size to get new IDs
        corder = np.argsort(blocksz)

        # Get new block labels, ordered by size
        newc = np.zeros(self.N)
        for c in np.arange(self.C):
            newc[self.c==corder[c]]=c

        # Get a permutation of the nodes to new class labels
        perm = np.argsort(-newc)
        return perm

    def plot(self, network, ax=None, color='k', F_true=None, lmbda_true=None):
        """
        """

        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, aspect="equal")

        perm = self.invariant_sbm_order()
        A = network.A[np.ix_(perm,perm)]
        W = network.W.sum(2)[np.ix_(perm, perm)]
        wlim = np.amax(abs(W))
        ax.imshow(A*W, interpolation="none", cmap="RdGy",
                  vmin=-wlim, vmax=wlim)

        return ax

    ###
    ### Implement Gibbs sampling for SBM
    ###
    def resample(self, A):
        self.resample_p([A])
        self.resample_c(A)
        self.resample_m()

    def resample_p(self, As):
        """
        Resample p given observations of the weights
        """
        def _get_mask(c1, c2):
            mask = ((self.c==c1)[:,None] * (self.c==c2)[None,:])
            if self.special_case_self_conns:
                mask &= True - np.eye(self.N, dtype=np.bool)
            return mask

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                mask = _get_mask(c1, c2)
                n_conns   = sum([A[mask].sum() for A in As])
                n_noconns = sum([(1 - A[mask]).sum() for A in As])

                tau1 = self.tau1 + n_conns
                tau0 = self.tau0 + n_noconns
                self.p[c1,c2] = np.random.beta(tau1, tau0)

        # Resample self connection probability
        if self.special_case_self_conns:
            mask = np.eye(self.N, dtype=np.bool)
            n_conns   = sum([A[mask].sum() for A in As])
            n_noconns = sum([(1 - A[mask]).sum() for A in As])

            tau1 = self.tau1 + n_conns
            tau0 = self.tau0 + n_noconns
            self.p_self = np.random.beta(tau1, tau0)

    def resample_c(self, A):
        """
        Resample block assignments given the weighted adjacency matrix
        and the impulse response fits (if used)
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

                    if n2 == n1:
                        # If we are special casing the self connections then
                        # we can just continue if n1==n2 since its weight has
                        # no bearing on the cluster assignment
                        if self.special_case_self_conns:
                            continue
                        else:
                            lp[cn1] += A[n1,n1] * np.log(self.p[cn1,cn1]) + \
                                       (1-A[n1,n1]) * np.log(1-self.p[cn1,cn1])

                    else:
                        # p(An1,n2] | c)
                        lp[cn1] += A[n1,n2] * np.log(self.p[cn1,cn2]) + \
                                   (1-A[n1,n2]) * np.log(1-self.p[cn1,cn2])

                        # p(A[n2,n1] | c)
                        lp[cn1] += A[n2,n1] * np.log(self.p[cn2,cn1]) + \
                                   (1-A[n2,n1]) * np.log(1-self.p[cn2,cn1])

            # Resample from lp
            self.c[n1] = sample_discrete_from_log(lp)

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)

