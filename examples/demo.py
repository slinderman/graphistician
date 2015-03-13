import scipy
import scipy.special
import matplotlib.pyplot as plt

from graphistician.internals.graph_models import *


def sample_er():
    """
    Test the generation and fitting of an Erdos Renyi model
    """
    N = 100
    a0 = 0.75
    b0 = 0.75
    model = ErdosRenyiNetwork(rho=None, x=(a0,b0))
    
    plt.figure()
    
    for ex in np.arange(3):
        plt.subplot(1,3,ex+1)
        (A,f,theta) = sample_network(model,N)
        plt.spy(A)
        plt.title("ER (rho=%.3f)" % theta)
    plt.show()
    
#    N_iter = 400
#    (f_trace, theta_trace, lp_trace) = fit_network(A, model, N_iter=N_iter)
#        
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(np.arange(N_iter), np.array(lp_trace))
#    plt.ylabel("log pr")
#    plt.title("log probability")
#    
#    plt.subplot(2,1,2)
#    plt.plot(np.arange(N_iter), np.array(theta_trace))
#    plt.hold(True)
#    plt.plot(np.arange(N_iter), theta*np.ones(N_iter))
#    plt.ylabel("\\theta")
#    plt.ylim(0,1)
#    plt.xlabel("Iteration")
#    plt.title("rho")
#    plt.show()

def sample_sbm():
    """
    Test the generation and fitting of an Erdos Renyi model
    """
    N = 100
    R = 5
    b1 = 0.5
    b0 = 0.5
    a  = 0.75
    alpha0 = a*np.ones(R)
    model = StochasticBlockModel(R, b0, b1, alpha0)
    
    def invariant_order(f):
        """
        Return an (almost) invariant ordering of the block labels 
        """
        # Cast features to block IDs
        z = np.array(f).astype(np.int)
        # Create a copy
        zc = np.copy(z)
        # Sort block IDs according to block size
        M = np.zeros(R)
        for r in np.arange(R):
            M[r] = np.sum(z==r)
        # Sort by size to get new IDs
        newz = np.argsort(M)
        # Update labels in zc
        for r in np.arange(R):
            zc[z==newz[r]]=r
        return np.argsort(-zc)
        
    # Generate a test network or use a given network
    plt.figure()
    for ex in np.arange(3):
        (A,f,theta) = sample_network(model,N)    
        zs = invariant_order(f)
        plt.subplot(1,3,ex-1)
        plt.spy(A[np.ix_(zs,zs)])
        plt.title("SBM")
    plt.show()

def gibbs_sbm((A,f)=(None,None)):
    """
    Test the generation and fitting of an Erdos Renyi model
    """
    R = 5
    b1 = 0.5
    b0 = 0.5
    a  = 1
    alpha0 = a*np.ones(R)
    model = StochasticBlockModel(R, b0, b1, alpha0)
        
    # Generate a test network or use a given network
    if A is None:
        N = 100
        (A,f,theta) = sample_network(model,N)    
    else:
        N = A.shape[0]
        f = np.ravel(f)
        (_,_,theta) = sample_network(model,N)
    
    print "lp model: %f" % model.logpr(A, f, theta)
    
    
    def invariant_order(f):
        """
        Return an (almost) invariant ordering of the block labels 
        """
        # Cast features to block IDs
        z = np.array(f).astype(np.int)
        # Create a copy
        zc = np.copy(z)
        # Sort block IDs according to block size
        M = np.zeros(R)
        for r in np.arange(R):
            M[r] = np.sum(z==r)
        # Sort by size to get new IDs
        newz = np.argsort(M)
        # Update labels in zc
        for r in np.arange(R):
            zc[z==newz[r]]=r
        return np.argsort(-zc)
            
    zs = invariant_order(f)
        
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.spy(A[np.ix_(zs,zs)])
    plt.title("True Network")
    
    
    ax2 = fig.add_subplot(1,2,2)
    h = plt.spy(A)
    plt.title("Inferred Network")
    plt.show(block=False)
    
    # Define a callback to update the inferred block structure
    def update_plot(f,theta):
        """
        Update the spy plot with the current block assignments
        """
        zs = invariant_order(f)
        h.set_data(A[np.ix_(zs,zs)])
        fig.canvas.draw()
     

    # Run the Gibbs sampler
    N_restarts = 1
    N_iter = 50
    plt.figure()
    for restart in np.arange(N_restarts):
        print "Restart %d" % restart
        (f_trace, theta_trace, lp_trace) = fit_network(A, model, 
                                                       N_iter=N_iter,
                                                       callback=update_plot)
        
        print "Gibbs sampler finished"
        plt.hold(True)
        plt.plot(np.arange(N_iter), np.array(lp_trace))
    
    plt.ylabel("log probability")
    plt.xlabel("Iteration")
    plt.title("Log probability for multiple Markov chains")
    plt.show()

def geweke_sbm_test():
    """
    Test our Gibbs sampler using Geweke validation
    """
    N = 10
    R = 5
    b1 = 1.5
    b0 = 0.5
    a  = 1.0
    alpha0 = a*np.ones(R)
    model = StochasticBlockModel(R, b0, b1, alpha0)
    
    # Program some bugs into the graph model
    # set_bugs(bug1,bug2,bug3)
    
    def invariant_order(f):
        """
        Return an (almost) invariant ordering of the block labels 
        """
        # Cast features to block IDs
        z = np.array(f).astype(np.int)
        # Create a copy
        zc = np.copy(z)
        # Sort block IDs according to block size
        M = np.zeros(R)
        for r in np.arange(R):
            M[r] = np.sum(z==r)
        # Sort by size to get new IDs
        newz = np.argsort(M)
        # Update labels in zc
        for r in np.arange(R):
            zc[z==newz[r]]=r
        return np.argsort(-zc)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    h = plt.spy(np.random.rand(N,N)<0.5)
    plt.title("Current network")
    plt.show(block=False)
    
    # Define a callback to update the inferred block structure
    def update_plot(A,f,theta):
        """zs
        Update the spy plot with the current block assignments
        """
        zs = invariant_order(f)
        h.set_data(A[np.ix_(zs,zs)])
        fig.canvas.draw()
             

    # Run the Gibbs sampler
    N_iter = 500
    (f_trace, theta_trace, lp_trace) = geweke_test(N, model, 
                                                   N_iter=N_iter,
                                                   callback=update_plot)
    
    print "Gibbs sampler finished"
    plt.figure()
    plt.plot(np.arange(N_iter), np.array(lp_trace))
    plt.ylabel("log pr")
    plt.title("log probability")
    plt.show(block=False)
    
 
    # Plot the trace of block assignments
    plt.figure()
    Z = np.array(f_trace)
#     plt.plot(np.arange(N_iter), Z[:,0])
#     plt.ylabel("z_0")
#     plt.title("z_0 trace")

    (pZ0, bins) = np.histogram(Z[:,0], 
                               np.arange(-0.5,R+0.5,1), 
                               density=True)
    
    plt.bar(bins[:-1],pZ0)
    plt.xlim(-1,R)
    plt.hold(True)
    xx = np.linspace(-1,R,100)
    plt.plot(xx,1.0/R*np.ones(100),'--r')
    plt.title('Empirical vs Predicted p(z[0])')
    plt.ylabel("p(z[0]=r)")
    plt.xlabel("r")
    plt.show(block=False)
    
    
    
    # Plot the trace of the pi's
    plt.figure()
    b00s = reduce(lambda a,(B,pi): a + [B[0,0]], theta_trace,[])
#     plt.plot(np.arange(N_iter),b00s)
#     plt.ylabel("B[0,0]")
#     plt.title("B[0,0] trace")
    N_bins = 25
    (pb00, bins) = np.histogram(b00s, np.linspace(0, 1, N_bins), density=True)    
    plt.bar(bins[:-1],pb00,1.0/N_bins)
    plt.hold(True)
    xx = np.linspace(0,1,100)
    prior_b00 = xx**(b1-1)*(1.0-xx)**(b0-1)/scipy.special.beta(b1,b0)
    plt.plot(xx,prior_b00,'r')
    plt.title('Empirical vs Predicted p(B[0,0])')
    plt.ylabel("p(B[0,0])")
    plt.xlabel("\rho")
    plt.show(block=False)
    
    raw_input("Press enter to quit.")

    
def load_data(fname):
    """
    Load a specific test network
    """
    import scipy.io
    dat = scipy.io.loadmat(fname)
    return dat

raw_input("Press enter to sample ER networks.")
sample_er()
raw_input("Press enter to sample SBM networks.")
sample_sbm()
raw_input("Press enter to fit SBM parameters with Gibbs sampling.")
gibbs_sbm()
raw_input("Press enter to perform Geweke validation on SBM Gibbs sampler.")
geweke_sbm_test()
