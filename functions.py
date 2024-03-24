import numpy as np


def sigmoid(x):
    """Shrinks a value between 0 and 1."""
    return 1 / (1 + np.exp(-x))


def random(*shape):
    """Random numbers between -1 and +1"""
    return np.random.uniform(-1, 1, shape)
