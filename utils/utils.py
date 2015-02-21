import numpy as np
from scipy.special import erfc, erfcinv

def normal_pdf(x, mu=0.0, sigma=1.0):
    z = (x-mu) / sigma
    return 1.0 / np.sqrt(2*np.pi) / sigma * np.exp(-0.5 * z**2)

def normal_cdf(x, mu=0.0, sigma=1.0):
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

def expected_truncnorm(mu=0, sigma=1.0, lb=-np.Inf, ub=np.Inf):
    """
    Compute the expected value of a truncated normal random variable.

    The only reason we don't use the scipy version is that we want to
    broadcast over arrays of mu, sigma, lb, and ub.

    We are using the form and notation from Wikipedia:
    http://en.wikipedia.org/wiki/Truncated_normal_distribution
    """
    # Broadcast arrays to be of the same shape
    mu, sigma, lb, ub = np.broadcast_arrays(mu, sigma, lb, ub)
    if np.allclose(sigma, 0.0):
        return mu

    # Compute the normalizer of the truncated normal
    Z = normal_cdf(ub) - normal_cdf(lb)

    # Standardize the bounds
    alpha = (lb-mu) / sigma
    beta  = (ub-mu) / sigma

    E = mu + (normal_pdf(alpha) - normal_pdf(beta)) / Z * sigma

    # # Initialize output
    # E = np.zeros_like(mu)
    #
    # # Break down into three cases
    # lower_bounded  = np.isfinite(lb) & np.isposinf(ub)
    # upper_bounded  = np.isfinite(ub) & np.isneginf(lb)
    # double_bounded = np.isfinite(lb) & np.isfinite(ub)
    # not_bounded    = np.isneginf(lb) & np.isposinf(ub)
    #
    # # Without bounds it reduces to standard normal
    # E[not_bounded] = mu[not_bounded]
    #
    # # TODO: Implement expectations when there is both an upper and lower bound
    # if np.any(double_bounded):
    #     raise NotImplementedError()
    #
    # # Compute lower bounded expectations
    # import pdb; pdb.set_trace()
    # alpha = (lb - mu)/sigma
    # lmbda = normal_pdf(alpha) / (1.0 - normal_cdf(alpha))
    # E[lower_bounded] = (mu + sigma * lmbda)[lower_bounded]
    #
    # # Compute upper bounded expectations
    # beta = (ub - mu)/sigma
    # lmbda = normal_pdf(beta) / normal_cdf(beta)
    # E[upper_bounded] = (mu - sigma * lmbda)[upper_bounded]

    return E

