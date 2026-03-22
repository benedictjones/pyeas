from __future__ import annotations

import math
import numpy as np

from typing import Any
from typing import cast, Union
from typing import Optional

from pyeas._population_scaling import PopScale

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMAES:
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import CMA

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = CMA(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
                trial_pop = optimizer.ask()
                solutions = []
                for trial in trial_pop:                 
                    value = quadratic(trial[0], trial[1])
                    solutions.append((x, value))
                    print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        start:
            Starting location (i.e., Initial mean vector of multi-variate gaussian distributions), select from:
                - 'mean', the center of the selected bounds,
                - 'random': initial random population used to select a good starting location,
                - passed in array 

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        groupings:
            An array which informs the object of the shape of a population member (optional).
                None        --> each member is a 1d array
                                e.g., possible member: [1.5, 0.5, 0.6, -0.9, 1.1]
                Otherwise   --> each member contains several diffenet shaped arrays 
                                e.g., groupings=[1,3,2] 
                                      possible member: [ [1.5], [0.5, 0.6, -0.9], [1.1]]

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).
    """

    def __init__(
        self,
        sigma: float,
        start: Union[np.ndarray, str] = 'mean',
        bounds: Optional[np.ndarray] = None,
        groupings: Optional[Union[np.ndarray, list]] = None,
        n_max_resampling: int = 20,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        # # Check number of dimensions
        if groupings is None:
            n_dim = len(bounds)
        else:
            n_dim = np.sum(groupings)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        # # Initialise object to normalise and denormalise the population
        self.PopScale = PopScale(np.array(bounds), groupings)
        normalised_bounds = np.array([[0,1] for b in range(n_dim)])  # set normalised bounds 

        # Checks on starting location (a.k.a starting mean)
        if isinstance(start, str):            
            if start == 'mean':
                self._mean = np.mean(normalised_bounds, axis=1)
            elif start == 'random':
                self._mean = None
            else:
                raise ValueError("Starting location can be: 'mean', 'random', or an array.")
        else:
            mean = self.PopScale._norm([start])[0]  # normalise mean
            assert np.all(
                np.abs(mean) < _MEAN_MAX
            ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

            self._mean = mean.copy()
            assert normalised_bounds is None or _is_valid_bounds(normalised_bounds, mean), "invalid bounds"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = np.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(
            weights_prime[mu:] ** 2
        )

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim**2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        

        if cov is None:
            self._C = np.eye(n_dim)
        else:
            assert cov.shape == (n_dim, n_dim), "Invalid shape of covariance matrix"
            self._C = cov

        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # bounds contains low and high of each parameter.
        self._bounds = normalised_bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        # self._rng = np.random.RandomState(seed)
        self._rng = np.random.default_rng(seed)
        self._trial_seed = self._rng.integers(10000, size=1)[0]


        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

        self._toggle = 0
        self._toggle_parent = 0
        self._parent = None
        self._parent_fit = None
        self._number_evals = 0  # number of training evaluations
        self._seed = seed

        self.history = {}
        self.history['best_fits'] = []
        self.history['best_solutions'] = []
        self.history['num_evals'] = []

        return

    def __getstate__(self) -> dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            # Remove _rng in pickle serialized object.
            if name == "_rng":
                continue
            if name == "_C":
                sym1d = _compress_symmetric(self._C)
                attrs["_c_1d"] = sym1d
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["_C"] = _decompress_symmetric(state["_c_1d"])
        del state["_c_1d"]
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        # setattr(self, "_rng", np.random.RandomState())  # legacy
        setattr(self, "_rng", np.random.default_rng(self._seed)) 

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    @property
    def parent(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent poulation"""
        return self.PopScale._denorm([self._mean])[0]

    @property
    def best_trial(self) -> np.ndarray:
        """Fetch the current psudo-population best member and it's fitness"""
        fit = np.min(self._trial_fits)
        member = self._trials[np.argmin(self._trial_fits)]
        return (fit, member)

    @property
    def best(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent poulation"""
        return (self.history['best_fits'][-1], self.history['best_solutions'][-1])

    @property
    def evals(self) -> int:
        """The number of evaluations (i.e., number of computations)"""
        return self._number_evals
    
    #

    # #########################################
    # # Create Population members to evaluate

    def ask(self) -> np.ndarray:
        """Sample a whole trial population which needs to be evaluated"""

        assert self._toggle == 0, "Must first evaluate current trials and tell me their fitnesses."
        assert self._toggle_parent == 0, "Must first evaluate and set the best/parent member fitness"

          # # Generate initital random population
        if self._mean is None:
            trial_pop = self._sample_random_initi_pop() # generate initial parent population to evaluate
        
        # # Sample Trial Population from Multivariate Gaussian Distribution (MGD)
        else:
            trial_pop = []
            for i in range(self._popsize):
                x = self.ask_member()
                trial_pop.append(x)

        self._toggle = 1
        return self.PopScale._denorm(trial_pop)


    def ask_member(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _sample_random_initi_pop(self) -> np.ndarray:
        """Sample a random initital normalised population"""
        norm_pop = []
        for i in range(self._popsize):
            norm_pop.append(np.around(self._rng.random(self._n_dim), decimals=5))
        return np.asarray(norm_pop)
    
    def _eigen_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        # z = self._rng.randn(self._n_dim)  # ~ N(0, I)  # legacy np.random.RandomState(seed)
        z = self._rng.standard_normal(self._n_dim)  # ~ N(0, I)  # newer np.random.default_rng(seed)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    #

    # #########################################
    # # Use the fed back fitnesses to perform a generational update

    def tell(self, fitnessess: list, trials: list) -> None:
        """Tell evaluation values"""

        assert len(fitnessess) == self._popsize, "Must tell popsize-length solutions."
        assert len(fitnessess) == self._popsize, "Must tell popsize-length solutions."
        assert self._toggle == 1, "Must first ask (i.e., fetch) & evaluate new trials."
        assert self._toggle_parent == 0, "Must first evaluate and set the best/parent member fitness"
        
        self._trials = trials  # allows the best tiral member property to access the lastest (real valued) members
        trials_norm = self.PopScale._norm(trials)
        self._trial_fits = fitnessess

        # # Set starting mean from initital random population
        if self._mean is None:
            mean = self.PopScale._norm([trials[np.argmin(fitnessess)]])[0]  # initial (normalised) mean value
            assert _is_valid_bounds(self._bounds, mean), "invalid bounds"
            self._mean = mean

        # # Update Multivariate Gaussian Distribution (MGD)
        else:
            solutions = list(zip(trials_norm, fitnessess))  # return to the solution format desired
            
            for s in solutions:
                assert np.all(
                    np.abs(s[0]) < _MEAN_MAX
                ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

            self._g += 1
            solutions.sort(key=lambda s: s[1])

            # Stores 'best' and 'worst' values of the
            # last 'self._funhist_term' generations.
            funhist_idx = 2 * (self.generation % self._funhist_term)
            self._funhist_values[funhist_idx] = solutions[0][1]
            self._funhist_values[funhist_idx + 1] = solutions[-1][1]

            # Sample new population of search_points, for k=1, ..., popsize
            B, D = self._eigen_decomposition()
            self._B, self._D = None, None

            x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
            y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

            # Selection and recombination
            y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
            self._mean += self._cm * self._sigma * y_w

            # Step-size control
            C_2 = cast(
                np.ndarray, cast(np.ndarray, B.dot(np.diag(1 / D))).dot(B.T)
            )  # C^(-1/2) = B D^(-1) B^T
            self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
                self._c_sigma * (2 - self._c_sigma) * self._mu_eff
            ) * C_2.dot(y_w)

            norm_p_sigma = np.linalg.norm(self._p_sigma)
            self._sigma *= np.exp(
                (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
            )
            self._sigma = min(self._sigma, _SIGMA_MAX)

            # Covariance matrix adaption
            h_sigma_cond_left = norm_p_sigma / math.sqrt(
                1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
            )
            h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
            h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

            # (eq.45)
            self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
                self._cc * (2 - self._cc) * self._mu_eff
            ) * y_w

            # (eq.46)
            w_io = self._weights * np.where(
                self._weights >= 0,
                1,
                self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
            )

            delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
            assert delta_h_sigma <= 1

            # (eq.47)
            rank_one = np.outer(self._pc, self._pc)
            rank_mu = np.sum(
                np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
            )
            self._C = (
                (
                    1
                    + self._c1 * delta_h_sigma
                    - self._c1
                    - self._cmu * np.sum(self._weights)
                )
                * self._C
                + self._c1 * rank_one
                + self._cmu * rank_mu
            )

        self._number_evals += len(fitnessess)  # number of training evaluations
        self.history['num_evals'].append(self._number_evals)

        self._toggle = 0
        self._toggle_parent = 1
        return

    def tellAgain(self, parent_fit: float):
        """Set the parent member's fitness"""

        # assert self._parent_fit is not None, "Can't set the best fitness until you have evaluated generation zero"
        assert self._toggle == 0, "Can only set the best result once you have updated the pop via a tell."
        assert self._toggle_parent == 1, "Must first perform a generational update, then inform about the new parent fit (e.g., optimizer.best = parent_fit)"
        # assert parent_fit >= 0, "The parent fitness score must be greater than zero."

        self._parent_fit = parent_fit
        
        # # Don't increase number of evaluations, as this doesn't progress evo
        # self._number_evals += 1

        self.history['best_fits'].append(parent_fit)
        self.history['best_solutions'].append(self.parent)

        self._toggle_parent = 0
        return

    #

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False

    #

    # #########################################
    # # Misc functions

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        self._seed
        return






def _is_valid_bounds(bounds: Optional[np.ndarray], mean: np.ndarray) -> bool:
    if bounds is None:
        return True
    if (mean.size, 2) != bounds.shape:
        print("\nBad bounds: mean starting value and bounds array have different shapes")
        return False
    if not np.all(bounds[:, 0] <= mean):
        print("\nBad bounds: mean below lower bound \n", mean)
        return False
    if not np.all(mean <= bounds[:, 1]):
        print("\nBad bounds: mean above upper bound \n", mean)
        return False
    return True


def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray:
    assert len(sym2d.shape) == 2 and sym2d.shape[0] == sym2d.shape[1]
    n = sym2d.shape[0]
    dim = (n * (n + 1)) // 2
    sym1d = np.zeros(dim)
    start = 0
    for i in range(n):
        sym1d[start : start + n - i] = sym2d[i][i:]  # noqa: E203
        start += n - i
    return sym1d


def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out

