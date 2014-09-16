import numpy as np

def invariant_sbm_order(f, R):
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