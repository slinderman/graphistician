"""
A set of compositions for connecting graph models
"""

class And(object):
    """
    Compose two network models by AND-ing them together.
    That is, multiply their probabilities of connection.
    """
