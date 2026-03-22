import warnings
import numpy as np
from numpy.testing import assert_almost_equal

def summation(a,b):
    return a+b


def test_summation():
    """
    Testing Summation function
    """
    assert summation(2, 10) == 12
    assert summation(3, 5) == 8
    assert summation(4, 6) == 10



            