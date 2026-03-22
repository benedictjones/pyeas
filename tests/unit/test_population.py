import numpy as np
from numpy.testing import assert_almost_equal
import warnings

from pyeas.population import Genes, Population



class TestPopulationInitialisation:

    member = [
        Genes(bounds=(-1,0), number=2),
        Genes(bounds=(4,8), number=1),
        Genes(bounds=(0,1), number=2),
    ]


    pop = Population(
        size=3,
        member=member,
    )

    def test_bounds(self):
        """
        Testing Summation function
        """
        assert np.shape(self.pop.bounds) == (5,2), 'Pop Bounds should have 2 limits'
        assert np.shape(self.pop.bounds)[0] == self.pop.n_dimensions, 'Pop Bounds should be equal to dim'



            