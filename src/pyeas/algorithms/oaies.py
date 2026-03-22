import numpy as np
import copy
from typing import Optional, Literal, Tuple, List, Any
import logging
from pydantic import BaseModel, Field, field_validator

from pyeas.population import Population
from pyeas.utils.boundary import handle_bound_violation
from pyeas.utils.tracking import HistoryEvals, HistoryTrials, HistoryBestParent, ensure_nd_array

logger = logging.getLogger(__name__)


class History(HistoryEvals, HistoryTrials, HistoryBestParent, BaseModel):
    """Data class to store OAIES results history"""
    gradient_steps: List[np.ndarray] = Field([], description="History of number of population member evaluations")
    
    def append_ga(self, ga:np.ndarray):
        self.gradient_steps = self.gradient_steps + [ga]

    @field_validator('gradient_steps', mode='after')
    @classmethod
    def ensure_gradient_steps(cls, value: Any) -> any:
        if len(value) > 0:
            for v in value:
                ensure_nd_array(v, 1)
        return value



class OAIES:
    """
    OpenAI-ES stochastic optimizer class with ask-and-tell interface.
    Based off the style of: https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py
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
    def parent(self) -> Optional[np.ndarray]:
        """Return the denormalised (and grouped) parent population"""
        if self._parent_norm is None:
            return None
        return self._population.denormalise(np.array([self._parent_norm]))[0]

    @property
    def best_member(self):
        """Fetch the current best solution and it's training fitness"""
        return self.history.best
    
    #

    def __init__(
        self,
        population: Population,
        alpha: float=0.001,
        sigma: float=0.05,
        optimiser: Literal['vanilla', 'momentum', 'adam'] = 'adam',
        constraint_handle: Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']] = 'clip',
        momentum: Optional[float] = None,
        beta1:float=0.9,
        beta2:float=0.999,
        seed: Optional[int] = None,
    ):
        """
        Initialise OpenAI-ES stochastic optimizer class with ask-and-tell interface.
        (Gradient are estimated)

        Args:
            population (Population): 
                Initialise Population object.
            alpha (float): 
                The learning rate or step size
                Defaults to 0.001
            sigma (float): 
                The gaussian noise used to perturb the normalized pseudo-population to create a trail population.
                Defaults to 0.05
            optimiser (Literal['vanilla', 'momentum', 'adam'], optional): 
                The gradient calculation method. 
                Defaults to 'adam'.
            constraint_handle (Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']], optional): 
                A string which assigns the method of handling boundary violations during mutation (optional).
                Schemes available: clip/projection, resample, scaled, reflection . 
                Defaults to 'reflection'.
            momentum (Optional[float], optional): _description_. Defaults to None.
            beta1 (float, optional): 
                 Decay rates for the moving averages of the gradient. 
                Defaults to 0.9.
            beta2 (float, optional): 
                 Decay rates for the moving averages of the squared gradient. 
                Defaults to 0.999.
            seed (Optional[int], optional): 
                Random seed. 
                Defaults to None.
        """
        self._population = copy.deepcopy(population)

        # # Make random generator object
        self._rng = np.random.default_rng(seed)
        self._trial_seed = self._rng.integers(10000, size=1)[0]


        # # Check other hyper-params
        if alpha <= 0:
            raise ValueError("The value of alpha (i.e., learning rate) must be larger than 0")
        if sigma < 0:
            raise ValueError("The value of sigma (i.e., the pseudo-population noise) must be larger than 0")

        self._alpha = alpha
        self._sigma = sigma

        if optimiser not in ['vanilla', 'momentum', 'adam']:
            raise ValueError("The selected optimiser should be a string (e.g., vanilla, momentum, adam)")
        self._optimiser = optimiser
        self._prev_ga = None 

        if self._optimiser == 'momentum':
            if momentum < 0 or momentum > 1:
                raise ValueError("Momentum must be [0,1]")
            self._m = momentum
        elif self._optimiser == 'adam':
            self._m = 0
            self._v = 0
            if beta1 <= 0:
                raise ValueError("The value of adam's beta1 must be larger than 0")
            if beta2 <= 0:
                raise ValueError("The value of adam's beta2 must be larger than 0")
            self._beta1 = beta1
            self._beta2 = beta2

    
        self._toggle = 0
        self._toggle_parent = 0

        self._parent_norm = None
        self._parent_fit = None

        self._constraint_handle = constraint_handle
        self._number_evals = 0  # number of training evaluations

        self.history = History()

        return

    #    


    # #########################################
    # # Create Population members to evaluate

    def _sample_trial_pop(self, loop:Optional[int]) -> np.ndarray:
        """Sample **normalised** trial population"""

        # # Create number generator for trial member (optionally include loop to allow repetability)
        seed = None
        if loop is not None:
            seed = self._trial_seed+loop
        trial_rng = np.random.default_rng(seed)

        # # Gen Gausian Pertubations To create trial psudo-population
        N = trial_rng.normal(size=(self.population.size, self.population.n_dimensions))

        trial_list = []
        for j in range(self.population.size):

            no_violations_left = False
            resample_count = 0
            while no_violations_left is False:
            
                # Create trial by adding noise
                mutant = self._parent_norm + self._sigma*N[j]

                 # If the mutants values violate the bounds, deal with it
                checked_mutant, no_violations_left = handle_bound_violation(mutant, handle=self._constraint_handle)
                resample_count += 1

                if resample_count >= 100:
                    checked_mutant, no_violations_left = handle_bound_violation(mutant, handle='clip')
                    break 

            trial_list.append(checked_mutant)
        
        trial_pop = np.asarray(trial_list, dtype=object)
        trial_pop = np.around(trial_pop.astype(float), decimals=5)

        return trial_pop

    def ask(self, loop: Optional[int] = None) -> np.ndarray:
        """Sample a whole trial population which needs to be evaluated"""

        if self._toggle != 0:
            raise ValueError("Must first evaluate current trials and tell me their fitnesses.")

        # # Generate population
        if self._parent_norm is None:
            self._toggle = 1
            _trial = self.population.population
        else:
            trial_pop = self._sample_trial_pop(loop)  # generate trial population to evaluate
            self._toggle = 1
            _trial = self.population.denormalise(trial_pop)

        self.history.append_trial(_trial)
        return _trial

    #

    # #########################################
    # # Use the fed back fitnesses to perform a generational update

    def tell(
        self, 
        fitnesses: list, 
        trials: Optional[np.ndarray] = None, 
        t: int=1
    ) -> None:
        """Tell the object the fitness values of the whole trial pseudo-population which has been valuated
            Args:

                fitnesses:
                    List of fitnesses for the corresponding trial members (i.e., pseudo-population)

                trials:
                    The trial members (i.e., pseudo-population) considered.

                t:
                    The iteration (optional). If not iterated, a static decay schedule is used.
        """

        # # if condition returns False, AssertionError is raised:
        if len(fitnesses) != self.population.size:
            raise ValueError("Must tell with popsize-length solutions.")
        if self._toggle != 1:
            raise ValueError("Must first ask (i.e., fetch) & evaluate new trials.")
        if t < 0:
            raise ValueError("The time/iteration must be greater than zero.")
        if self._toggle_parent != 0:
            raise ValueError("Must first evaluate and set the best/parent member fitness")

        self._trials = trials
        self._trial_fits = fitnesses
        trials_norm = self.population.normalise(trials)

        # # Retrieve fitness information and make gradient decent update
        if self._parent_fit is None:
            # # Use a uniformly generated population to select a starting location
            self._parent_norm = np.copy(trials_norm[np.argmin(fitnesses)])

            self._parent_fit = np.min(fitnesses) 
            self.history.append_ga(np.full(np.shape(self._parent_norm), np.nan))

        else:
            
            # Collapse the pseudo-population to update the parent/target
            if trials is None:
                raise ValueError("To perfom GD, please tell me the fitnesses and trials/pseudo-population used")
            
            # parent member to perform GD on
            theta = np.copy(self._parent_norm)  

            std = 1e-8
            R = -np.array(fitnesses)
            if np.std(R) > 0:
                std = np.std(R)
            A = (R - np.mean(R)) / std

            # # Recall: mutant = self._parent_norm + self._sigma*N[j]
            # _epsilon = trials_norm # use trials
            # _epsilon = trials_norm - self._parent_norm # use  trial_perterbations            
            _epsilon = (trials_norm - self._parent_norm)/self._sigma # use pertubations only
            


            # # Grad Estimate
            g = 1/(self.population.size*self._sigma) * np.dot(_epsilon.T, A)

            # # Normal Grad Decent
            if self._optimiser == 'vanilla':
                ga = g*self._alpha

            # # Momentum
            elif self._optimiser == 'momentum':
                # https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
                if self._prev_ga is None:
                    ga = g*self._alpha
                else:
                    ga = g*self._alpha + self._m*self._prev_ga
                self._prev_ga = ga

            # # Adam GD
            elif self._optimiser == 'adam':
                # https://machinelearningmastery.com/adam-optimization-from-scratch/
                # https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
                self._m = self._beta1*self._m + (1-self._beta1)*g
                self._v = self._beta2*self._v + (1-self._beta2)*g**2

                m_hat = self._m/(1-self._beta1**t)
                v_hat = self._v/(1-self._beta2**t)

                ga = self._alpha*m_hat/(1e-8+v_hat**0.5)
                

            else:
                raise ValueError("Invalid gradient decent optimiser method")

            # Update from new grad step
            theta = theta + ga

            self.history.append_ga(ga.astype(float))
            # logging.info(f"\n[OAIES] - ga (step size): {np.around(ga.astype(float), decimals=7)}")
            theta = np.around(theta.astype(float), decimals=8)
            theta, _ = handle_bound_violation(theta, handle='clip')
            self._parent_norm = theta
            # print("theta:", theta)
            # exit()
        
        self._number_evals += len(trials)
        self.history.append_evals(self._number_evals)

        self._toggle = 0
        self._toggle_parent = 1
        return
    
    def tell_parent(self, parent_fit: float):
        """Set the parent member's fitness"""

        if self._parent_fit is None:
            raise ValueError("Can't set the best fitness until you have evaluated generation zero")
        if self._toggle != 0:
            raise ValueError("Can only set the best result once you have updated the pop via a tell.")
        if self._toggle_parent != 1:
            raise ValueError("Must first perform a generational update, then inform about the new parent fit (e.g., optimizer.best = parent_fit)")

        self._parent_fit = parent_fit

        self.history.append_parent(
            self.parent,
            self._parent_fit,
        )

        self._toggle_parent = 0
        return
        
    #

    # #########################################
    # # Misc functions

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        return
    
    #

    # # fin
    
    