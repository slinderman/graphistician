import numpy as np
from scipy.special import erfc, erfcinv

def normal_cdf(x, mu, sigma):
    z = (x-mu)/sigma
    return 0.5 * erfc(-z/ np.sqrt(2))

def sample_truncnorm(mu=0, sigma=1, lb=-np.Inf, ub=np.Inf):
    """ Sample a truncated normal with the specified params
    """
    # Broadcast arrays to be of the same shape
    mu, sigma, lb, ub = np.broadcast_arrays(mu, sigma, lb, ub)
    shp = mu.shape
    if np.allclose(sigma, 0.0):
        return mu

    cdflb = normal_cdf(lb, mu, sigma)
    cdfub = normal_cdf(ub, mu, sigma)

    # Sample uniformly from the CDF
    cdfsamples = cdflb + np.random.rand(*shp)*(cdfub-cdflb)

    # Clip the CDF samples so that we can invert them
    cdfsamples = np.clip(cdfsamples, 1e-15, 1-1e-15)

    zs = -np.sqrt(2) * erfcinv(2*cdfsamples)

    assert np.all(np.isfinite(zs))

    return sigma * zs + mu
