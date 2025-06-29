import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from pyeas._population import Genes, Population


member = [
    Genes(bounds=(-1,0), number=2),
    Genes(bounds=(4,8), number=1),
    Genes(bounds=(0,1), number=2),
]


pop = Population(
    size=3,
    member=member,
)

def test_bounds():
    """
    Testing Summation function
    """
    assert np.shape(pop.bounds)[1] == 2, 'Pop Bounds should have 2 limits'
    assert np.shape(pop.bounds)[1] == pop.n_dimensions, 'Pop Bounds should be equal to dim'



            