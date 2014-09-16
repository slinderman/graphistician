"""
A few simple graph model classes that fall into the Aldous-Hoover framework.

Scott Linderman
11/7/2013
"""

import numpy as np
from scipy.special import betaln
import matplotlib.pyplot as plt

import string 
import logging
import copy

from hips.inference.discrete_sample import discrete_sample
from hips.inference.log_sum_exp_sample import log_sum_exp_sample
from hips.inference.elliptical_slice import elliptical_slice
from hips.inference.hmc import hmc

class AldousHooverNetwork:
    """
    Base class for Aldous-Hoover random graphs
    """
    def __init__(self, A=None):
        """
        Initialize a random Aldous-Hoover random graph
        """
        pass
        
    def pr_A_given_f(self,f1,f2,theta):
        """
        Compute the probability of an edge from a node with 
        features f1 to a node with features f2, given global parameters
        theta.
        """
        raise Exception("pr_A_given_f is not implemented!")
    
    def logpr_f(self,f,theta):
        """
        Compute the prior probability of feature f.
        """
        raise Exception("logpr_f is not implemented!")
    
    def logpr_theta(self,theta):
        """
        Compute the prior probability of parameters theta.
        """
        raise Exception("logpr_theta is not implemented!")

    def log_lkhd(self, A, f, theta):
        ll = 0
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[0]):
                ll += A[i,j]*np.log(self.pr_A_given_f(f[i], f[j], theta)) + \
                      (1-A[i,j])*np.log(1-self.pr_A_given_f(f[i], f[j], theta))
        return ll

    def logpr(self, A, f, theta, beta=1.0):
        """
        Compute the log probability of a network given the
        node features, the parameters, and the adjacency matrix A.

        :param beta: The weight of the log likelihood
        """
        lp = self.logpr_theta(theta)

        for fi in f:
            lp += self.logpr_f(fi, theta)

        lp += beta * self.log_lkhd(A, f, theta)

        return lp

    def sample_f(self, theta, (n,A,f)=(None,None,None), beta=1.0):
        """
        Sample a set of features. If n,A, and f are given, 
        sample the features of the n-th node from the posterior 
        having observed A and the other features.

        :param beta: The weight of the log likelihood
        """
        raise Exception("sample f is not implemented!")
    
    def sample_theta(self, (A,f)=(None,None), beta=1.0):
        """
        Sample the parameters of pr_A_given_f. If A and f
        are given, sample these parameters from the posterior
        distribution.

        :param beta: The weight of the log likelihood
        """
        raise Exception("sample_theta is not implemented!")
    
    def sample_A(self, f, theta):
        """
        Sample A given features f and parameters theta
        """ 
        N = len(f)
        A = np.zeros((N,N))
        for i in np.arange(N):
            for j in np.arange(N):
                A[i,j] = self.sample_Aij(f[i],f[j],theta)
        
        return A
    
    def sample_Aij(self, fi, fj, theta):
        """
        Sample a single entry in the network
        """
        return np.random.rand() < self.pr_A_given_f(fi,fj, theta)

    
class ErdosRenyiNetwork(AldousHooverNetwork):
    """
    Model an Erdos-Renyi random graph
    """
    def __init__(self, rho=None, x=None):
        """ Constructor.
        :param rho    sparsity of the graph (probability of an edge)
        :param (a,b)  parameters of a Beta prior on rho
        
        Either rho or (a,b) must be specified
        """
#        super(ErdosRenyiNetwork,self).__init__()
        if rho is None:
            if x is None:
                raise Exception("Either rho or (a,b) must be specified")
            else:
                (a,b) = x
        else:
            if x is not None:
                raise Exception("Either rho or (a,b) must be specified")
            else:
                a = None
                b = None
                        
        self.rho = rho
        self.a = a
        self.b = b
        
    def pr_A_given_f(self,fi,fj, theta):
        """
        The probability of an edge is simply theta, regardless
        of the "features" fi and fj.
        """
        rho = theta
        return rho
        
    def logpr_f(self,f,theta):
        """
        There are no features for the Erdos-Renyi graph
        """
        return 0.0
    
    def logpr_theta(self, theta):
        """
        This varies depending on whether or not rho is specified
        """
        rho = theta
        if self.rho is not None:
            if self.rho == rho:
                return 0.0
            else:
                return -np.Inf
            
        else:
            return (self.a-1.0)*np.log(rho) + (self.b-1.0)*np.log((1.0-rho)) - \
                   betaln(self.a,self.b)
    
    def sample_f(self, theta, (n,A,f)=(None,None,None), beta=1.0):
        """
        There are no features in the Erdos-Renyi graph model.
        """
        return None
    
    def sample_theta(self, (A,f)=(None,None), beta=1.0):
        """
        Sample the parameters of pr_A_given_f. For the Erdos-Renyi
        graph model, the only parameter is rho.
        """
        if self.rho is not None:
            return self.rho
        elif A is None and f is None:
            return np.random.beta(self.a,self.b)
        else:
            N = A.shape[0]
            nnz_A = np.sum(A)
            a_post = self.a + beta * nnz_A
            b_post = self.b +  beta * (N**2 - nnz_A)
            return np.random.beta(a_post, b_post)
            
class StochasticBlockModel(AldousHooverNetwork):
    """
    Model an Erdos-Renyi random graph
    """
    def __init__(self, R, b0, b1, alpha0):
        """ Constructor.
        :param R     Number of blocks
        :param b0    prior probability of edge
        :param b1    prior probability of no edge
        :param alpha prior probability of block membership
        
        Either rho or (a,b) must be specified
        """
#        super(ErdosRenyiNetwork,self).__init__()
        self.R = R
        self.b0 = b0
        self.b1 = b1
        self.alpha0 = alpha0
                
    def pr_A_given_f(self,fi,fj, theta):
        """
        The probability of an edge is beta distributed given 
        the blocks fi and fj
        """
        zi = fi
        zj = fj
        (B,pi) = theta
        return B[zi,zj]
        
    def logpr_f(self,f,theta):
        """
        The features of a stochastic block model are the nodes'
        block affiliations
        """
        (B,pi) = theta
        fint = np.array(f).astype(np.int)
        lp = 0.0
        lp += np.sum(np.log(pi[fint])) 
        return lp
    
    def logpr_theta(self,theta):
        """
        This varies depending on whether or not rho is specified
        """
        (B,pi) = theta
        lp = 0.0
        
        # Add prior on B
        for ri in np.arange(self.R):
            for rj in np.arange(self.R):
                lp += (self.b1-1.0)*np.log(B[ri,rj]) + (self.b0-1.0)*np.log((1.0-B[ri,rj])) - \
                       betaln(self.b1,self.b0)
        
        # Add prior on pi
        for ri in np.arange(self.R):
            lp += np.sum((self.alpha0-1.0)*np.log(pi))
            
        return lp
    
    def sample_f(self, theta, (n,A,f)=(None,None,None), beta=1.0):
        """
        Sample new block assignments given the parameters.

        :param beta: The weight of the log likelihood
        """
        (B,pi) = theta
        if n is None and A is None and f is None:
            # Sample the prior
            zn = discrete_sample(pi)
        else:
            # Sample the conditional distribution on f[n]
            zn = self.naive_sample_f(theta, n, A, f, beta=beta)
#            zn = self.collapsed_sample_f(theta, n, A, f)
    
        return zn
    
    def naive_sample_f(self, theta, n, A, f, beta=1.0):
        """
        Naively Gibbs sample z given B and pi
        """
        (B,pi) = theta
        A = A.astype(np.bool)
        zother = np.array(f).astype(np.int)
        
        # Compute the posterior distribution over blocks
        ln_pi_post = np.log(pi)
        
        rrange = np.arange(self.R)
        for r in rrange:
            zother[n] = r
            # Block IDs of nodes we connect to 
            o1 = A[n,:]
            if np.any(A[n,:]):
                ln_pi_post[r] += beta * np.sum(np.log(B[np.ix_([r],zother[o1])]))
            
            # Block IDs of nodes we don't connect to
            o2 = np.logical_not(A[n,:])
            if np.any(o2):
                ln_pi_post[r] += beta * np.sum(np.log(1-B[np.ix_([r],zother[o2])]))
            
            # Block IDs of nodes that connect to us
            i1 = A[:,n]
            if np.any(i1):
                ln_pi_post[r] += beta * np.sum(np.log(B[np.ix_(zother[i1],[r])]))

            # Block IDs of nodes that do not connect to us
            i2 = np.logical_not(A[:,n])
            if np.any(i2):
                ln_pi_post[r] += beta * np.sum(np.log(1-B[np.ix_(zother[i2],[r])]))
            
        zn = log_sum_exp_sample(ln_pi_post)
        
        return zn
    
    def collapsed_sample_f(self, theta,n,A,f):
        """
        Use a collapsed Gibbs sampler to update the block assignments 
        by integrating out the block-to-block connection probabilities B.
        Since this is a Beta-Bernoulli model the posterior can be computed
        in closed form and the integral can be computed analytically.
        """
        (B,pi) = theta
        A = A.astype(np.bool)
        zother = np.array(f).astype(np.int)
        
        # P(A|z) \propto 
        #    \prod_{r1}\prod_{r2} Beta(m(r1,r2)+b1,\hat{m}(r1,r2)+b0) /
        #                           Beta(b1,b0)
        # 
        # Switching z changes the product over r1 and the product over r2
        
        # Compute the posterior distribution over blocks
        
        # TODO: This literal translation of the log prob is O(R^3)
        # But it can almost certainly be sped up to O(R^2)
        ln_pi_post = np.log(pi)
        for r in np.arange(self.R):
            zother[n] = r
            for r1 in np.arange(self.R):
                for r2 in np.arange(self.R):
                    # Look at outgoing edges under z[n] = r
                    Ar1r2 = A[np.ix_(zother==r1,zother==r2)]
                    mr1r2 = np.sum(Ar1r2)
                    hat_mr1r2 = Ar1r2.size - mr1r2
                    
                    ln_pi_post[r] += betaln(mr1r2+self.b1, hat_mr1r2+self.b0) - \
                                     betaln(self.b1,self.b0)
                                
            zn = log_sum_exp_sample(ln_pi_post)
        
        return zn
        
    def sample_theta(self, (A,f)=(None,None), beta=1.0):
        """
        Sample the parameters of pr_A_given_f. For the Erdos-Renyi
        graph model, the only parameter is rho.

        :param beta: The weight of the log likelihood
        """
        if A is None and f is None:
            # Sample B and pi from the prior
            B = np.random.beta(self.b1,self.b0,
                               (self.R,self.R))
            pi = np.random.dirichlet(self.alpha0)
        else:
            # Sample pi from its Dirichlet posterior
            z = np.array(f).astype(np.int)
            alpha_post = np.zeros((self.R,))
            for r in np.arange(self.R):
                alpha_post[r] = self.alpha0[r] + np.sum(z==r)
            pi = np.random.dirichlet(alpha_post)
            
            # Sample B from its Beta posterior
            B = np.zeros((self.R,self.R), dtype=np.float32)
            for r1 in np.arange(self.R):
                for r2 in np.arange(self.R):
                    b0post = self.b0
                    b1post = self.b1
                    
                    Ar1r2 = A[np.ix_(z==r1, z==r2)]
                    if np.size(Ar1r2) > 0:
                        b0post += beta * np.sum(1-Ar1r2)
                        b1post += beta * np.sum(Ar1r2)
                    
                    B[r1,r2] = np.random.beta(b1post, b0post)
        
        return (B,pi)
                
# Define helper functions to sample a random graph
def sample_network(model, N):
    """
    Sample a new network with N nodes from the given model.
    """
    # First sample theta, the parameters of the base measure
    theta = model.sample_theta()
    
    # Then sample features for each node
    f = []
    for n in np.arange(N):
        f.append(model.sample_f(theta))
                
    # Finally sample the network itself
    A = model.sample_A(f,theta)
    
    return (A,f,theta)

def fit_network(A, model, x0=None, N_iter=1000, callback=None, pause=False):
    """
    Fit the parameters of the network model using MCMC.
    """
    N = A.shape[0]
    
    # If the initial features are not specified, start with a 
    # draw from the prior.
    if x0 is None:
        theta0 = model.sample_theta()
        
        f0 = []
        for n in np.arange(N):
            f0.append(model.sample_f(theta0))
            
    else:
        (f0,theta0) = x0       

    print "Starting Gibbs sampler"    
    f = copy.deepcopy(f0)
    theta = copy.deepcopy(theta0)
    
    lp_trace = np.zeros(N_iter)
    f_trace = []
    theta_trace = []
    for iter in np.arange(N_iter):
        lp = model.logpr(A,f,theta)
        lp_trace[iter] = lp
        
        print "Iteration %d. \tlog pr: %f" % (iter, lp_trace[iter])
        
        # Sample the model parameters theta
        theta = model.sample_theta((A,f))
        
        # Sample features f
        for n in np.arange(N):
            f[n] = model.sample_f(theta, (n,A,f))
        
        # If the user supplied a callback, call it now
        if callback is not None:
            callback(f, theta)
        
        f_trace.append(f)
        theta_trace.append(theta)
        
        if pause:
            raw_input("Press enter to continue.")
    
    return (f_trace, theta_trace, lp_trace)


def geweke_test(N, model, N_iter=1000, callback=None, pause=False):
    """
    Fit the parameters of the network model using MCMC.
    """    
    # If the initial features are not specified, start with a 
    # draw from the prior.
    theta0 = model.sample_theta()
    
    f0 = []
    for n in np.arange(N):
        f0.append(model.sample_f(theta0))
          
    A0 = model.sample_A(f0, theta0)

    print "Starting Gibbs sampler"    
    f = copy.deepcopy(f0)
    theta = copy.deepcopy(theta0)
    A = np.copy(A0)
    
    lp_trace = np.zeros(N_iter)
    f_trace = []
    theta_trace = []
    A_trace = []
    for iter in np.arange(N_iter):
        lp = model.logpr(A,f,theta)
        lp_trace[iter] = lp
        
        print "Iteration %d. \tlog pr: %f" % (iter, lp_trace[iter])
        
        # Sample the model parameters theta
        theta = model.sample_theta((A,f))
        
        # Sample features f
        for n in np.arange(N):
            f[n] = model.sample_f(theta, (n,A,f))
        
        # Sample a new graph A given the updated features
        A = model.sample_A(f, theta)
        
        # If the user supplied a callback, call it now
        if callback is not None:
            callback(A, f, theta)
        
        # Save a copy of the data
        theta_trace.append(copy.deepcopy(theta))
        f_trace.append(copy.deepcopy(f))
        A_trace.append(A.copy())
        
        if pause:
            raw_input("Press enter to continue.")
    
    return (f_trace, theta_trace, lp_trace)
