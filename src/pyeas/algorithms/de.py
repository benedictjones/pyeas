import numpy as np
import copy
from typing import Optional, List, Union, Dict, Any, Literal, Tuple
import time
import logging 
from pydantic import BaseModel

from pyeas.population import Population
from pyeas.utils.boundary import handle_bound_violation
from pyeas.utils.tracking import HistoryEvals, HistoryTrials, HistoryBestParent, Solution

logger = logging.getLogger(__name__)


class History(HistoryEvals, HistoryTrials, HistoryBestParent, BaseModel):
    """Data class to store DE results history"""
    a:Optional[int] = None

class DE:
    """Differential Evolution (DE) stochastic optimizer class with ask-and-tell interface.

    > based off the style of: https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py

    (Build into a package: https://www.youtube.com/watch?v=5KEObONUkik)

    """


    # #########################################
    # # Properties: https://www.freecodecamp.org/news/python-property-decorator/ 

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self.history.generation
    
    @property
    def population(self):
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._population
    
    @property
    def parent_pop(self) -> np.ndarray:
        """Return the denormalized (and grouped) parent population"""
        return self._population.population

    @parent_pop.setter
    def parent_pop(self, new_parent_pop: np.ndarray):
        """Set the parent population"""
        self._population.population = new_parent_pop
        return

    @property
    def best_member(self) -> Tuple[float, np.ndarray]:
        """Fetch the current best member and it's training fitness"""
        return Solution(
            member=self.population.population[self._best_idx],
            loss=self._pop_fits[self._best_idx],
        )
    
    #

    def __init__(
        self,
        population: Population,
        mut: float,
        crossp: float,
        mut_scheme: Literal['rand1', 'best1', 'rand2', 'best2', 'ttb1'] = 'best1',
        constraint_handle: Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']] = 'reflection',
        seed: Optional[int] = None,
    ):
        """
        Initialise the DE object.

        Args:
            population (Population): 
                Initialise Population object.
            mut (float): 
                Mutation Factor (i.e., 'F') for selected mutation scheme.
            crossp (float): 
                Crossover Rate (i.e., 'CR') for binary crossover.            
            mut_scheme (Literal['rand1', 'best1', 'rand2', 'best2', 'ttb1'], optional): 
                A string which assigns the mutation scheme used (optional).
                Schemes available: best1, best2, rand1, rand2, ttb1 (target-to-best). 
                Defaults to 'best1'.
            constraint_handle (Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']], optional): 
                A string which assigns the method of handling boundary violations during mutation (optional).
                Schemes available: clip/projection, resample, scaled, reflection . 
                Defaults to 'reflection'.
            seed (Optional[int], optional): 
                Random state. 
                Defaults to None.
        """
        
        self._population = copy.deepcopy(population)

        # # Make random generator object
        self._rng = np.random.default_rng(seed)
        self._trial_seed = self._rng.integers(10000, size=1)[0]


        # # Check other hyper-params
        if mut < 0:
            raise ValueError("The value of mutation factor (i.e., F) must be larger than 0")
        self._mut = mut

        if isinstance(mut_scheme, str) is False:
            raise ValueError("The mutation scheme (e.g., best1, rand1) must be a string")
        self._mut_scheme = mut_scheme

        if crossp < 0 or crossp > 1:
            raise ValueError("The value of crossover factor (i.e., CR or crossp) must be [0,1]")
        self._crossp = crossp
        
        
        self._toggle = 0

        self._pop_fits = None
        self._best_idx = None
        self._number_evals = 0  # number of training evaluations
        self._constraint_handle = constraint_handle

        self.history = History()

        return

    

    # #########################################
    # # Create Population members to evaluate

    def ask(self, loop: Optional[int] = None) -> np.ndarray:
        """Sample a whole trial population which needs to be evaluated"""

        if self._toggle != 0:
            raise ValueError("Must first evaluate current trials and tell me their fitnesses.")

        # # Generate population
        if self._pop_fits is None:
            self._toggle = 1
            _trial = self.population.population
        else:
            trial_pop = self._sample_trial_pop(loop)  # generate trial population to evaluate
            self._toggle = 1
            _trial = self.population.denormalise(trial_pop)

        self.history.append_trial(_trial)
        return _trial

    def _sample_trial_pop(self, loop:Optional[int]) -> np.ndarray:
        """Sample trial normalised population"""

        trial_list = []
        for j in range(self.population.size):

            # # Create number generator for trial member (optionally include loop to allow repetability)
            seed = None
            if loop is not None:
                seed = self._trial_seed+loop+j
            trial_rng = np.random.default_rng(seed)

            # # Select Indexes to generate mutants from
            """
            Creates an range array(0, popsize) but excludes the current
            value of j, used to randomly select pop involved in mutation.
            i.e idxs is all pop index's except the current one
            """
            idxs = [idx for idx in range(self.population.size) if idx != j]

            # # Mutation
            mutant = self._mutate(idxs, j, trial_rng)

            # # Recombination & Replacement
            trial = self._bin_cross(j, trial_rng, mutant)

            trial_list.append(trial)
        
        trial_pop = np.asarray(trial_list, dtype=object)
        trial_pop = np.around(trial_pop.astype(np.float64), decimals=5)

        return trial_pop

    def _mutate(
        self, 
        idxs:list, 
        current_idx:int, 
        trial_rng:np.random.default_rng,
    ):
        """
        Selects which mutation scheme to use, and returns the mutant.
        """
        no_violations_left = False
        resample_count = 0
        while no_violations_left is False:

            if self._mut_scheme == 'rand1':
                mutant = self._rand1(idxs, trial_rng)

            elif self._mut_scheme == 'best1':
                mutant = self._best1(idxs, trial_rng)

            elif self._mut_scheme == 'rand2':
                mutant = self._rand2(idxs, trial_rng)

            elif self._mut_scheme == 'best2':
                mutant = self._best2(idxs, trial_rng)

            elif self._mut_scheme == 'ttb1':
                mutant = self._ttb1(idxs, trial_rng, current_idx)

            else:
                raise ValueError("Invalit Mutation Scheme: %s" % (self._mut_scheme))

            # If the mutants values violate the bounds, deal with it
            mutant, no_violations_left = handle_bound_violation(mutant, handle=self._constraint_handle)
            resample_count += 1

            if resample_count >= 100:
                mutant, no_violations_left = handle_bound_violation(mutant, handle='clip')
                break
        return mutant

    #

    # # mutation methods

    def _rand1(self, idxs: list, trial_rng: object) -> np.ndarray:
        """
        Random1 mutation method.
        Randomly choose 3 indexes without replacement.
        """
        # selected = np.random.choice(idxs, 3, replace=False)
        selected = trial_rng.choice(idxs, 3, replace=False)
        np_pop = np.asarray(self.population.population_raw, dtype=object)
        a, b, c = np_pop[selected]  # assign to a variable
        # note this is not the real pop values

        # mutant
        mutant = a + self._mut * (b - c)

        return mutant  # this is unformatted

    #

    def _rand2(self, idxs: list, trial_rng: object) -> np.ndarray:
        """
        Random2 mutation method.
        Randomly choose 5 indexes without replacement
        """

        # selected = np.random.choice(idxs, 5, replace=False)
        selected = trial_rng.choice(idxs, 5, replace=False)
        np_pop = np.asarray(self.population.population_raw, dtype=object)
        a, b, c, d, e = np_pop[selected]  # assign to a variable
        # note; a, b etc are genomes

        # mutant
        mutant = a + self._mut * (b - c + d - e)

        return mutant  # this is unformatted

    #

    def _best1(self, idxs: list, trial_rng: object) -> np.ndarray:
        """
        Best1 mutation method
        Randomly choose 2 indexes without replacement, combined with the best.
        """

        # selected = np.random.choice(idxs, 2, replace=False)
        selected = trial_rng.choice(idxs, 2, replace=False)
        np_pop = np.asarray(self.population.population_raw, dtype=object)
        b, c = np_pop[selected]
        a = self.population.population_raw[self._best_idx]

        # mutant
        mutant = a + self._mut * (b - c)

        return mutant  # this is unformatted

    #

    def _best2(self, idxs: list, trial_rng: object) -> np.ndarray:
        """
        Best2 mutation method
        Randomly choose 4 indexes without replacement, combined with the best.
        """

        # selected = np.random.choice(idxs, 4, replace=False)
        selected = trial_rng.choice(idxs, 4, replace=False)
        np_pop = np.asarray(self.population.population_raw, dtype=object)
        b, c, d, e = np_pop[selected]
        a = self.population.population_raw[self._best_idx]

        # mutant
        mutant = a + self._mut * (b - c + d - e)

        return mutant 
    #

    def _ttb1(self, idxs: list, trial_rng: object, current_idx: int) -> np.ndarray:
        """
        Target-to-best/1 mutation method
        Randomly choose 4 indexes without replacement, combined with the best.
        """

        # selected = np.random.choice(idxs, 2, replace=False)
        selected = trial_rng.choice(idxs, 2, replace=False)
        np_pop = np.asarray(self.population.population_raw, dtype=object)
        a = self.population.population_raw[current_idx]
        b = self.population.population_raw[self._best_idx]
        c, d = np_pop[selected]

        F1 = self._mut
        F2 = self._mut

        # mutant
        mutant = a + F1 * (b - a) + F2 * (c - d)

        return mutant  
    
    #

    # # Recombination & Replacement

    def _bin_cross(self, j, trial_rng, mutant):
        """
        Basic binary crossover
        """

        # # Return true or false for each of the random elements
        # cross_points = np.random.rand(self.population.n_dimensions) < cr
        cross_points = trial_rng.random(size=self.population.n_dimensions) < self._crossp

        # # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.population.n_dimensions)
        x = int(trial_rng.choice(a))

        cross_points[x] = True

        # # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, self.population.population_raw[j])

        return trial

    #

    # #########################################
    # # Use the fed back fitnesses to perform a generational update

    def tell(
        self, 
        fitnesses: list, 
        trials: Optional[np.ndarray] = None
    ) -> None:
        """Tell the object the fitness values of the whole trial population which has been valuated"""

        # # if condition returns False, AssertionError is raised:
        if len(fitnesses) != self.population.size:
            raise ValueError("Must tell with popsize-length solutions.")
        if self._toggle != 1:
            raise ValueError("Must first ask (i.e., fetch) & evaluate new trials.")


        # # Retrieve fitness information and make population update
        if self._pop_fits is None:
            self._pop_fits = np.array(fitnesses)  # assign the initi pop fits
            self._best_idx = np.argmin(fitnesses)  # assign the best initi pop index
            
        else:
            # evaluate trial population to evaluate
            # # Compare the children/trial genomes to the parent/target
            new_pop = np.copy(self.population.population_raw)
            new_pop_fits = np.copy(self._pop_fits)
            if trials is None:
                raise ValueError("To update the population, please tell me the fitnesses and trials used")
            trials = self.population.normalise(trials)
            old_pop_best_fit = self.best_member.loss


            for j in range(self.population.size):
                #print("\ncompare fi", j, "fi=", fitnesses[j], "to previous fitness=", self._pop_fits[j], " prev best:", old_pop_best_fit)

                # # find best index
                if fitnesses[j] <= new_pop_fits[self._best_idx]:
                    #print("  best update:", j, " prev best:", self._pop_fits[self._best_idx])
                    self._best_idx = j  

                # # whether to keep parent or child/trial
                if fitnesses[j] <= self._pop_fits[j]:  
                    #print("  pop update:", j)
                    new_pop[j] = trials[j] 
                    new_pop_fits[j] = fitnesses[j]

            # # Quick Check 
            if old_pop_best_fit < new_pop_fits[self._best_idx]:
                raise ValueError("wrong! old best: %f, new best: %f, best idx: %d" % (old_pop_best_fit, new_pop_fits[self._best_idx], self._best_idx))
            
            # # Assign updated population as the parent 
            self.population.population_raw = new_pop
            self._pop_fits = new_pop_fits

        self._number_evals += len(trials)

        self.history.append_parent(
            self.best_member.member,
            self.best_member.loss,
        )
        self.history.append_evals(self._number_evals)

        self._toggle = 0
        return
    
    #

    # #########################################
    # # Misc functions

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        return
    
    #

    # # fin
    
    