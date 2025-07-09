import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase

from pyeas._de import DE


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2




class TestDE(TestCase):
    def test_runs(self):
        
        optimizer = DE(mut=[0.4],
                       
               crossp=0.4,
               bounds=np.array([[-10,10],[-20,20]]),
               population_size=10)

        for generation in range(50):
            solutions = []
            
            # Ask a parameter
            trial_pop = optimizer.ask()

            for trial in trial_pop:
                value = quadratic(trial[0], trial[1])
                solutions.append((trial, value))
                print(f"#{generation} {value} (x1={trial[0]}, x2 = {trial[1]})")

            # Tell evaluation values.
            optimizer.tell(solutions)




            