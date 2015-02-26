"""
Network models expose a probability of connection and a scale of the weights
"""
import abc
import copy
import numpy as np
from scipy.special import psi
from scipy.misc import logsumexp

from abstractions import GaussianWeightedNetworkDistribution, NetworkDistribution, \
    WeightedDirectedNetwork, FixedGaussianNetwork
from internals.deps.pybasicbayes.util.stats import sample_discrete_from_log
from internals.distributions import Bernoulli, Beta, Dirichlet, Discrete
from internals.weights import GaussianWeights


class _StochasticBlockModelBase(NetworkDistribution):
    """
    A stochastic block model is a clustered network model with
    K:          Number of nodes in the network
    C:          Number of blocks
    m[c]:       Probability that a node belongs block c
    p[c,c']:    Probability of connection from node in block c to node in block c'
    v[c,c']:    Scale of the gamma weight distribution from node in block c to node in block c'

    It is parameterized by:
    pi:         Parameter of Dirichlet prior over m
    tau0, tau1: Parameters of beta prior over p
    alpha:      Shape parameter of gamma prior over v
    beta:       Scale parameter of gamma prior over v
    """
    __metaclass__ = abc.ABCMeta

    # Override this in base classes!
    _weight_class = None
    _default_weight_hypers = {}

    def __init__(self, N, B,
                 C=1,
                 c=None, m=None, pi=1.0,
                 p=None, tau0=1.0, tau1=1.0,
                 weight_hypers={},
                 allow_self_connections=True):
        """
        Initialize SBM with parameters defined above.
        """
        self.N = N
        self.B = B

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


        # Initialize the weight models for each pair of blocks
        self.weight_hypers = copy.deepcopy(self._default_weight_hypers)
        self.weight_hypers.update(weight_hypers)
        self.weight_models = \
            [[self._weight_class(self.B,
                                 **self.weight_hypers)
              for c in range(self.C)]
             for c in range(self.C)]

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

    # def log_likelihood(self, x):
    #     """
    #     Compute the log likelihood of a set of SBM parameters
    #
    #     :param x:    (m,p,v) tuple
    #     :return:
    #     """
    #     m,p,v,c = x
    #
    #     lp = 0
    #     lp += Dirichlet(self.pi).log_probability(m)
    #     lp += Beta(self.tau1 * np.ones((self.C, self.C)),
    #                self.tau0 * np.ones((self.C, self.C))).log_probability(p).sum()
    #     lp += Gamma(self.mu0, self.Sigma0).log_probability(v).sum()
    #     lp += (np.log(m)[c]).sum()
    #     return lp
    #
    # def log_probability(self):
    #     return self.log_likelihood((self.m, self._p, self.v, self.c))

    def rvs(self, size=[]):
        # Sample a network given m, c, p, weight distributions
        A = np.zeros((self.N, self.N))
        W = np.zeros((self.N, self.N, self.B))

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                blk = (self.c==c1)[:,None] * (self.c==c2)[None,:]
                A[blk] = np.random.rand(blk.sum()) < self.p[c1,c2]
                W[blk,:] = self.weight_models[c1][c2].rvs(size=blk.sum())

        return FixedGaussianNetwork(A,W)

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

class _GibbsSBM(_StochasticBlockModelBase):
    """
    Implement Gibbs sampling for SBM
    """
    def resample(self, networks=[]):

        if not isinstance(networks, list):
            networks = [networks]

        # TODO: Handle multiple networks
        assert len(networks) == 1
        network = networks[0]

        # As = [n.A for n in networks]
        # Ws = [n.W for n in networks]
        A = network.A
        W = network.W

        # Resample weight models
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                c1_and_c2 = (self.c==c1)[:,None] * (self.c==c2)[None, :]
                subnet = FixedGaussianNetwork(A * c1_and_c2, W)
                # Wc1c2 = np.vstack([W[c1_and_c2, :] for W in Ws])
                self.weight_models[c1][c2].resample(subnet)

        self.resample_p([A])
        self.resample_c(networks[0])
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

    def resample_c(self, network):
        """
        Resample block assignments given the weighted adjacency matrix
        and the impulse response fits (if used)
        """
        if self.C == 1:
            return

        A = network.A
        W = network.W

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

                        # p(W[k,k'] | c)
                        lp[cn1] += (A[n1,n2] * self.weight_models[cn1][cn2]
                                   .log_likelihood(W[n1,n2,:])).sum()

                        lp[cn1] += (A[n2, n1] * self.weight_models[cn2][cn1]
                                   .log_likelihood(W[n2,n1,:])).sum()
                    else:
                        # Self connection
                        lp[cn1] += Bernoulli(self.p[cn1, cn1])\
                                        .log_probability(A[n1,n1]).sum()

                        lp[cn1] += (A[n1, n1] * self.weight_models[cn1][cn1]
                                       .log_likelihood(W[n1,n1,:])).sum()

            # Resample from lp
            self.c[n1] = sample_discrete_from_log(lp)

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)


class _MeanFieldSBM(_StochasticBlockModelBase):
    """
    Add mean field updates
    """
    def __init__(self, N, B,
                 **kwargs):

        super(_MeanFieldSBM, self).__init__(N, B, **kwargs)

        # Initialize mean field parameters
        self.mf_pi     = np.ones(self.C)

        # To break symmetry, start with a sample of mf_m
        self.mf_m      = np.random.dirichlet(10 * np.ones(self.C), size=(self.N,))
        self.mf_tau0   = self.tau0  * np.ones((self.C, self.C))
        self.mf_tau1   = self.tau1  * np.ones((self.C, self.C))

    # Get expectations from the weight model
    def meanfieldupdate(self, network):
        assert isinstance(network, WeightedDirectedNetwork)

        # Update the remaining SBM parameters
        self.mf_update_p(network)
        # Update the block assignments
        self.mf_update_c(network)
        # Update the probability of each block
        self.mf_update_m()
        # Update the weight models
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                pc1c2 = self.mf_m[:,c1][:,None] * self.mf_m[:,c2][None, :]
                self.weight_models[c1][c2].meanfieldupdate(network, weights=pc1c2)

    def meanfield_sgdstep(self, network, minibatchfrac, stepsize):
        # Update the remaining SBM parameters
        self.mf_update_p(network, stepsize=stepsize)
        self.mf_update_m(stepsize=stepsize)
        self.mf_update_c(network, stepsize=stepsize)

        # Update the weight models
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                pc1c2 = self.mf_m[:,c1][:,None] * self.mf_m[:,c2][None, :]
                self.weight_models[c1][c2].meanfieldupdate(network, pc1c2, stepsize=stepsize)

    def mf_update_m(self, stepsize=1.0):
        """
        Mean field update of the block probabilities
        :return:
        """
        pi_hat = self.pi + self.mf_m.sum(axis=0)
        self.mf_pi = (1.0 - stepsize) * self.mf_pi + stepsize * pi_hat

    def mf_update_p(self, network, stepsize=1.0):
        """
        Mean field update for the CxC matrix of block connection probabilities
        :param network: network to update based on
        :return:
        """
        E_A = network.E_A
        E_notA =  1 - E_A
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]


                if self.allow_self_connections:
                    tau1_hat = self.tau1 + (pc1c2 * E_A).sum()
                    tau0_hat = self.tau0 + (pc1c2 * E_notA).sum()
                else:
                    # TODO: Account for self connections
                    tau1_hat = self.tau1 + (pc1c2 * E_A).sum()
                    tau0_hat = self.tau0 + (pc1c2 * E_notA).sum()

                self.mf_tau1[c1,c2] = (1.0 - stepsize) * self.mf_tau1[c1,c2] + stepsize * tau1_hat
                self.mf_tau0[c1,c2] = (1.0 - stepsize) * self.mf_tau0[c1,c2] + stepsize * tau0_hat


    def mf_update_c(self, network, stepsize=1.0):
        """
        Update the block assignment probabilitlies one at a time.
        This one involves a number of not-so-friendly expectations.
        :return:
        """
        E_A    = network.E_A
        E_notA = 1 - network.E_A
        # Sample each assignment in order
        for n1 in xrange(self.N):
            # Compute unnormalized log probs of each connection
            lp = np.zeros(self.C)

            # Prior from m
            lp += self.mf_expected_log_m()

            # Iterate over possible block assignments
            for cn1 in xrange(self.C):

                # Likelihood from each edge in the network
                for n2 in xrange(self.N):
                    for cn2 in xrange(self.C):
                        pcn2 = self.mf_m[n2, cn2]

                        p_pn1n2 = Beta(self.mf_tau1[cn1,cn2], self.mf_tau0[cn1, cn2])
                        E_ln_p_n1n2 = p_pn1n2.expected_log_p()
                        E_ln_notp_n1n2 = p_pn1n2.expected_log_notp()
                        lp[cn1] += pcn2 * Bernoulli().negentropy(E_x=E_A[n1, n2],
                                                                 E_notx=E_notA[n1, n2],
                                                                 E_ln_p=E_ln_p_n1n2,
                                                                 E_ln_notp=E_ln_notp_n1n2)

                        # Compute the expected log likelihood of the weights
                        # Compute E[ln p(W | A=1, c)]
                        lp[cn1] += E_A[n1, n2] * pcn2 * self._expected_log_likelihood_W(network, n1,cn1,n2,cn2)

                    # Now do the same thing for the reverse edge
                    if n2 != n1:
                        p_pn2n1 = Beta(self.mf_tau1[cn2,cn1], self.mf_tau0[cn2, cn1])
                        E_ln_p_n2n1 = p_pn2n1.expected_log_p()
                        E_ln_notp_n2n1 = p_pn2n1.expected_log_notp()
                        lp[cn1] += pcn2 * Bernoulli().negentropy(E_x=E_A[n2, n1],
                                                                 E_notx=E_notA[n2, n1],
                                                                 E_ln_p=E_ln_p_n2n1,
                                                                 E_ln_notp=E_ln_notp_n2n1)

                        lp[cn1] += E_A[n2, n1] * pcn2 * self._expected_log_likelihood_W(network, n2,cn2,n1,cn1)


            # Normalize the log probabilities to update mf_m
            Z = logsumexp(lp)
            mk_hat = np.exp(lp - Z)

            self.mf_m[n1,:] = (1.0 - stepsize) * self.mf_m[n1,:] + stepsize * mk_hat

    @abc.abstractmethod
    def _expected_log_likelihood_W(self, network, n1, cn1, n2, cn2):
        """
        Compute the expected log likelihood of W[n1,n2]
        :param n1:
        :param n2:
        :return:
        """
        raise NotImplementedError()

    def mf_expected_p(self):
        """
        Compute the expected probability of a connection, averaging over c
        :return:
        """
        E_p = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_p += pc1c2 * self.mf_tau1[c1,c2] / (self.mf_tau0[c1,c2] + self.mf_tau1[c1,c2])

        if not self.allow_self_connections:
            np.fill_diagonal(E_p, 0.0)

        return E_p

    def mf_expected_notp(self):
        """
        Compute the expected probability of NO connection, averaging over c
        :return:
        """
        return 1.0 - self.mf_expected_p()

    def mf_expected_log_p(self):
        """
        Compute the expected log probability of a connection, averaging over c
        :return:
        """
        E_ln_p = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_p += pc1c2 * Beta(self.mf_tau1[c1,c2], self.mf_tau0[c1,c2]).expected_log_p()

        if not self.allow_self_connections:
            np.fill_diagonal(E_ln_p, -np.inf)

        return E_ln_p

    def mf_expected_log_notp(self):
        """
        Compute the expected log probability of NO connection, averaging over c
        :return:
        """
        E_ln_notp = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_notp += pc1c2 * Beta(self.mf_tau1[c1,c2],
                                          self.mf_tau0[c1,c2]).expected_log_notp()

        if not self.allow_self_connections:
            np.fill_diagonal(E_ln_notp, 0.0)

        return E_ln_notp

    def mf_expected_m(self):
        return self.mf_pi / self.mf_pi.sum()

    def mf_expected_log_m(self):
        """
        Compute the expected log probability of each block
        :return:
        """
        E_log_m = psi(self.mf_pi) - psi(self.mf_pi.sum())
        return E_log_m

    def expected_log_likelihood(self, network):
        raise NotImplementedError()

    def get_vlb(self):
        vlb = 0

        # Get the VLB of the expected class assignments
        E_ln_m = self.mf_expected_log_m()
        for n in xrange(self.N):
            # Add the cross entropy of p(c | m)
            vlb += Discrete().negentropy(E_x=self.mf_m[n,:], E_ln_p=E_ln_m)

            # Subtract the negative entropy of q(c)
            vlb -= Discrete(self.mf_m[n,:]).negentropy()

        # Get the VLB of the connection probability matrix
        # Add the cross entropy of p(p | tau1, tau0)
        vlb += Beta(self.tau1, self.tau0).\
            negentropy(E_ln_p=(psi(self.mf_tau1) - psi(self.mf_tau0 + self.mf_tau1)),
                       E_ln_notp=(psi(self.mf_tau0) - psi(self.mf_tau0 + self.mf_tau1))).sum()

        # Subtract the negative entropy of q(p)
        vlb -= Beta(self.mf_tau1, self.mf_tau0).negentropy().sum()

        # Get the VLB of the block probability vector, m
        # Add the cross entropy of p(m | pi)
        vlb += Dirichlet(self.pi).negentropy(E_ln_g=self.mf_expected_log_m())

        # Subtract the negative entropy of q(m)
        vlb -= Dirichlet(self.mf_pi).negentropy()

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                vlb += self.weight_models[c1][c2].get_vlb()

        return vlb

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        self.m = np.random.dirichlet(self.mf_pi)
        self.p = np.random.beta(self.mf_tau1, self.mf_tau0)

        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                self.weight_models[c1][c2].resample_from_mf()

        self.c = np.zeros(self.K, dtype=np.int)
        for k in xrange(self.K):
            self.c[k] = int(np.random.choice(self.C, p=self.mf_m[k,:]))

    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        raise NotImplementedError()

class GaussianStochasticBlockModel(_GibbsSBM, _MeanFieldSBM, GaussianWeightedNetworkDistribution):

    _weight_class = GaussianWeights
    _default_weight_hypers = {}

    @property
    def Mu(self):
        """
        Get the NxNxB array of mean weights
        :return:
        """
        Mu = np.zeros((self.N, self.N, self.B))
        for n1 in xrange(self.N):
            c1 = self.c[n1]
            for n2 in xrange(self.N):
                c2 = self.c[n2]
                Mu[n1,n2,:] = self.weight_models[c1][c2].mu
        return Mu

    @property
    def Sigma(self):
        """
        Get the NxNxB array of mean weights
        :return:
        """
        S = np.zeros((self.N, self.N, self.B))
        for n1 in xrange(self.N):
            c1 = self.c[n1]
            for n2 in xrange(self.N):
                c2 = self.c[n2]
                S[n1,n2,:,:] = self.weight_models[c1][c2].sigma
        return S

    # Mean field likelihood for updates
    def _expected_log_likelihood_W(self, network, n1, cn1, n2, cn2):
        """
        Compute the expected log likelihood of W[n1,n2]
        :param n1:
        :param n2:
        :return:
        """
        E_W = network.E_W
        E_WWT = network.E_WWT

        return self.weight_models[cn1][cn2].\
            expected_log_likelihood((E_W[n1,n2,:],
                                     E_WWT[n1,n2,:,:]))

    # Mean field expectations
    def mf_expected_mu(self):
        E_mu = np.zeros((self.N, self.N, self.B))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                E_mu += self.mf_m[:,c1][:,None,None] * self.mf_m[:,c2][None,:,None] * \
                        self.weight_models[c1][c2].mf_expected_mu()[None, None, :]
        return E_mu

    def mf_expected_mumuT(self):
        E_mumuT = np.zeros((self.N, self.N, self.B, self.B))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                E_mumuT += self.mf_m[:,c1][:,None,None,None] * self.mf_m[:,c2][None,:,None,None] * \
                           self.weight_models[c1][c2].mf_expected_mumuT()[None, None, :, :]
        return E_mumuT

    def mf_expected_Sigma_inv(self):
        E_Sigma_inv = np.zeros((self.N, self.N, self.B, self.B))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                E_Sigma_inv += self.mf_m[:,c1][:,None,None,None] * self.mf_m[:,c2][None,:,None,None] * \
                               self.weight_models[c1][c2].mf_expected_Sigma_inv()[None, None, :, :]

        return E_Sigma_inv

    def mf_expected_logdet_Sigma(self):
        E_logdet_Sigma = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                E_logdet_Sigma += self.mf_m[:,c1][:,None] * self.mf_m[:,c2][None,:] * \
                                  self.weight_models[c1][c2].mf_expected_logdet_Sigma()

        return E_logdet_Sigma
