"""
Super simple adjacency models. We either have a Bernoulli model
with fixed probability or a beta-Bernoulli model.
"""
import numpy as np

from abstractions import AdjacencyDistribution

class BernoulliAdjacencyDistribution(AdjacencyDistribution):
    """
    Bernoulli edge model with fixed probability
    """
    def __init__(self, N, p):
        super(BernoulliAdjacencyDistribution, self).__init__(N)

        assert p > 0 and p < 1
        self.p = p
        self._P = p * np.ones(N,N)

    @property
    def P(self):
        return self._P

    def log_prior(self):
        return 0

    def resample(self, edges):
        pass


class BetaBernoulliAdjacencyDistribution(AdjacencyDistribution):
    def __init__(self, N, tau1, tau0):
        super(BetaBernoulliAdjacencyDistribution, self).__init__(N)
        assert tau1 > 0 and tau0 > 0
        self.tau1 = tau1
        self.tau0 = tau0

        self.p = np.random.beta(tau1, tau0)

    @property
    def P(self):
        return self.p * np.ones((self.N, self.N))

    def rvs(self, size=[]):
        return np.random.rand(*size) < self.p

    def log_prior(self):
        return

    def resample(self, A):
        """
        Resample p given observations of the weights
        """
        n_conns = A.sum()
        n_noconns = A.size - n_conns

        tau1 = self.tau1 + n_conns
        tau0 = self.tau0 + n_noconns

        self.p = np.random.beta(tau1, tau0)


class LatentDistanceAdjacencyDistribution(object):
    """
    l_n ~ N(0, sigma^2 I)
    A_{n', n} ~ Bern(\sigma(-||l_{n'} - l_{n}||_2^2))
    """
    def __init__(self, N, dim=2, sigma=1.0, mu0=0.0,
                 allow_self_connections=True):
        self.N = N
        self.dim = dim
        self.sigma = sigma
        self.mu0 = mu0
        self.L = np.sqrt(self.sigma) * np.random.randn(N,dim)
        self.allow_self_connections = allow_self_connections

    @property
    def D(self):
        Mu = self.mu0 + -((self.L[:,None,:] - self.L[None,:,:])**2).sum(2)

        return Mu

    @property
    def P(self):
        P = logistic(self.D)

        if not self.allow_self_connections:
            np.fill_diagonal(P, 1e-32)

        return logistic(self.D)

    def log_prior(self):
        """
        Compute the prior probability of F, mu0, and lmbda
        """
        lp  = 0

        # Log prior of F under spherical Gaussian prior
        lp += -0.5 * (self.L * self.L / self.sigma).sum()
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

    def _hmc_log_probability(self, L, mu0, A):
        """
        Compute the log probability as a function of L.
        This allows us to take the gradients wrt L using autograd.
        :param L:
        :param A:
        :return:
        """
        # Compute pairwise distance
        L1 = np.reshape(L,(self.N,1,self.D))
        L2 = np.reshape(L,(1,self.N,self.D))
        D = - np.sum((L1-L2)**2, axis=2)

        # Compute the logit probability
        logit_P = mu0 + D

        # Take the logistic of the negative distance
        # P = 1.0 / (1+np.exp(logit_P))
        P = logistic(logit_P)

        # Compute the log likelihood
        ll = np.sum(A * np.log(P) + (1-A) * np.log(1-P))

        # Log prior of L under spherical Gaussian prior
        lp = -0.5 * np.sum(L * L / self.sigma)

        # Log prior of mu0 under standardGaussian prior
        lp += -0.5 * self.mu0**2

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

        # TODO: Handle self connections
        # np.fill_diagonal(A, False)

        return A

    def compute_optimal_rotation(self, L_true):
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
        R = orthogonal_procrustes(self.L, L_true)[0]
        return R

    def plot(self, A, ax=None, color='k', L_true=None, lmbda_true=None):
        """
        If D==2, plot the embedded nodes and the connections between them

        :param L_true:  If given, rotate the inferred features to match F_true
        :return:
        """

        import matplotlib.pyplot as plt

        assert self.D==2, "Can only plot for D==2"

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

        # Resample the offset
        self._resample_mu0(A)

    def _resample_L(self, A):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc

        lp  = lambda L: self._hmc_log_probability(L, self.mu0, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        self.L = hmc(lp, dlp, stepsz, nsteps, self.L.copy(), negative_log_prob=False)

    def _resample_mu0(self, A):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc


        lp  = lambda mu0: self._hmc_log_probability(self.L, mu0, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        mu0 = hmc(lp, dlp, stepsz, nsteps, np.array(self.mu0), negative_log_prob=False)
        self.mu0 = float(mu0)



class SBMAdjacencyDistribution(AdjacencyDistribution):
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
                 C=1,
                 c=None, m=None, pi=1.0,
                 p=None, tau0=1.0, tau1=1.0,
                 allow_self_connections=True):
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

    @property
    def P(self):
        """
        Get the KxK matrix of probabilities
        :return:
        """
        P = self.p[np.ix_(self.c, self.c)]
        if not self.allow_self_connections:
            np.fill_diagonal(P, 0.0)
        return P

    def log_prior(self):
        """
        Compute the log likelihood of a set of SBM parameters

        :param x:    (m,p,v) tuple
        :return:
        """


        lp = 0
        lp += Dirichlet(self.pi).log_probability(self.m)
        lp += Beta(self.tau1 * np.ones((self.C, self.C)),
                   self.tau0 * np.ones((self.C, self.C))).\
            log_probability(self.p).sum()
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
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):

                n_conns   = sum([A[np.ix_(self.c==c1, self.c==c2)].sum()
                                 for A in As])
                n_noconns = sum([(1 - A[np.ix_(self.c==c1, self.c==c2)]).sum()
                                 for A in As])

                if not self.allow_self_connections:
                    # TODO: Account for self connections
                    pass

                tau1 = self.tau1 + n_conns
                tau0 = self.tau0 + n_noconns
                self.p[c1,c2] = np.random.beta(tau1, tau0)

    def resample_c(self, A):
        """
        Resample block assignments given the weighted adjacency matrix
        and the impulse response fits (if used)
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

                    if n2 != n1:
                        # p(A[k,k'] | c)
                        lp[cn1] += Bernoulli(self.p[cn1, cn2])\
                                        .log_probability(A[n1,n2]).sum()

                        # p(A[k',k] | c)
                        lp[cn1] += Bernoulli(self.p[cn2, cn1])\
                                        .log_probability(A[n2,n1]).sum()

                    else:
                        # Self connection
                        lp[cn1] += Bernoulli(self.p[cn1, cn1])\
                                        .log_probability(A[n1,n1]).sum()

            # Resample from lp
            self.c[n1] = sample_discrete_from_log(lp)

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)

