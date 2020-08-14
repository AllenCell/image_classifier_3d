import numpy as np


def test_dummy():
    a = np.zeros((4, 4, 4))
    assert a.shape[0] > 0
