import numpy as np
import copy
from typing import Optional, List, Union, Dict, Any, Literal
import time

from pyeas._population import Population

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
        return len(self.history['best_fits'])
    
    @property
    def population(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._population
    
    @property
    def parent_pop(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent population"""
        return self._population.population

    @parent_pop.setter
    def parent_pop(self, new_parent_pop: np.ndarray):
        """Set the parent population"""
        self._population.population = new_parent_pop
        return

    @property
    def best_member(self) -> int:
        """Fetch the current best member and it's training fitness"""
        fit = self._pop_fits[self._best_idx]
        member = self.population.population[self._best_idx]
        return (fit, member)
    
    @property
    def evals(self) -> int:
        """The number of evaluations (i.e., number of computations)"""
        return self._number_evals
    
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

        if isinstance(constraint_handle, str) is False:
            raise ValueError("The mutation boundary handle constrain (e.g., clip, reflection) must be a string")
        self._constraint_handle = constraint_handle

        if crossp < 0 or crossp > 1:
            raise ValueError("The value of crossover factor (i.e., CR or crossp) must be [0,1]")
        self._crossp = crossp
        
        
        self._toggle = 0

        self._pop_fits = None
        self._best_idx = None
        self._best_idx = None
        self._number_evals = 0  # number of training evaluations

        self.history = {}
        self.history['best_fits'] = []
        self.history['best_solutions'] = []
        self.history['num_evals'] = []

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
            return self.population.population
        else:
            trial_pop = self._sample_trial_pop(loop)  # generate trial population to evaluate
            self._toggle = 1
            return self.population.denormalise(trial_pop)
    

    def _sample_trial_pop(self, loop) -> np.ndarray:
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

    #

    def _mutate(
            self, 
            idxs:list, 
            current_idx:int, 
            trial_rng:np.random.default_rng,
        ):
        """
        Selects which mutation scheme to use, and returns the mutant.
        """
        reinit = 1
        resample_count = 0
        while reinit == 1:

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
            mutant, reinit = self._handle_bound_violation(mutant)

            resample_count += 1

            if resample_count >= 100:
                mutant, reinit = self._handle_bound_violation(mutant, force_select='reflection')

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

    # # If a mutants value falls outide of the bounds, sort it out somehow

    def _handle_bound_violation(self, mutant, force_select=0):

        num_violations = self._count_bound_violation(mutant)

        # # if no violations, just return
        if num_violations == 0:
            return mutant, 0

        # # Implement the selected violation handeling sheme
        if force_select == 0:
            handle = self._constraint_handle
        else:
            handle = force_select

        #

        # No handling
        if handle is None:
            return mutant, 0
        
        # Perform projection (i.e., clipping)
        elif handle == 'clip' or handle == 'projection':
            mutant = np.clip(mutant, 0, 1)
            return mutant, 0

        # Return the resample flag
        elif handle == 'resample':
            return mutant, 1

        # Perform Scaled Mutant operation
        elif handle == 'scaled':
            alphas = [1]
            for m in mutant:
                if m > 1:
                    alphas.append(1/m)

            mutant = mutant*np.min(alphas)

            # # Not fool proof
            mutant = np.clip(mutant, 0, 1)

            return mutant, 0

        # Perform Scaled Mutant operation
        elif handle == 'reflection':
            for i, m in enumerate(mutant):
                if m > 1:
                    mutant[i] = 2-m
                elif m < 0:
                    mutant[i] = -m
                else:
                    mutant[i] = m

            return mutant, 0
        
        else:
            raise ValueError("The constraint_handle that selects how to manage boundary violations is not valid")

    def _count_bound_violation(self, mutant):
        num_violations = np.size(np.where(mutant < 0)) + np.size(np.where(mutant > 1))  # How many clips are there?
        # print("Num clips below 0: %d, Num clips above 1: %d" % (np.size(np.where(mutant < 0)), np.size(np.where(mutant > 1))))
        return num_violations
    
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

    def tell(self, fitnessess: list, trials: Optional[np.ndarray] = None) -> None:
        """Tell the object the fitness values of the whole trial population which has been valuated"""

        # # if condition returns False, AssertionError is raised:
        assert len(fitnessess) == self.population.size, "Must tell with popsize-length solutions."
        assert self._toggle == 1, "Must first ask (i.e., fetch) & evaluate new trials."


        # # Retrieve fitness information and make population update
        if self._pop_fits is None:
            self._pop_fits = np.array(fitnessess)  # assign the initi pop fits
            self._best_idx = np.argmin(fitnessess)  # assign the best initi pop index
            
        else:
            # evaluate trial population to evaluate
            # # Compare the children/trial genomes to the parent/target
            new_pop = np.copy(self.population.population_raw)
            new_pop_fits = np.copy(self._pop_fits)
            assert trials is not None, "To update the population, please tell me the fitnesses and trials used"
            trials = self.population.normalise(trials)
            old_best = self.best_member[0]


            for j in range(self.population.size):
                #print("\ncompare fi", j, "fi=", fitnessess[j], "to previous fitness=", self._pop_fits[j], " prev best:", old_best)

                # # find best index
                if fitnessess[j] <= new_pop_fits[self._best_idx]:
                    #print("  best update:", j, " prev best:", self._pop_fits[self._best_idx])
                    self._best_idx = j  

                # # whether to keep parent or child/trial
                if fitnessess[j] <= self._pop_fits[j]:  
                    #print("  pop update:", j)
                    new_pop[j] = trials[j] 
                    new_pop_fits[j] = fitnessess[j]

            # # Quick Check 
            assert old_best >= new_pop_fits[self._best_idx], "wrong! old best: %f, new best: %f, best idx: %d" % (old_best, new_pop_fits[self._best_idx], self._best_idx)
            
            # # Assign updated population as the parent 
            self.population.population_raw = new_pop
            self._pop_fits = new_pop_fits

        self._number_evals += len(trials)

        self.history['best_fits'].append(self.best_member[0])
        self.history['best_solutions'].append(self.best_member[1])
        self.history['num_evals'].append(self._number_evals)

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
    
    