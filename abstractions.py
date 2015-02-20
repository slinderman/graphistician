import numpy as np
import abc

from deps.pybasicbayes.abstractions import Distribution

class AldousHooverNetwork(Distribution):
    """
    Base class for Aldous-Hoover random graphs. These graphs are characterized
    by the property that the edges are conditionally independent given a set
    of per-node and per-edge random variables.
    """
    __metaclass__ = abc.ABCMeta
