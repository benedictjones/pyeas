import numpy as np 
from typing import Optional, Literal, Tuple


def count_bound_violation(mutant):
    num_violations = np.size(np.where(mutant < 0)) + np.size(np.where(mutant > 1))  # How many clips are there?
    return num_violations


def handle_bound_violation(
        mutant: np.ndarray, 
        handle: Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']] = 'reflection',
) -> Tuple[np.ndarray, bool]:

    num_violations = count_bound_violation(mutant)

    # # if no violations, just return
    if num_violations == 0:
        return mutant, False


    # No handling
    if handle is None:
        return mutant, False
    
    # Perform projection (i.e., clipping)
    elif handle == 'clip' or handle == 'projection':
        mutant = np.clip(mutant, 0, 1)
        return mutant, False

    # Return the resample flag
    elif handle == 'resample':
        return mutant, True

    # Perform Scaled Mutant operation
    elif handle == 'scaled':
        alphas = [1]
        for m in mutant:
            if m > 1:
                alphas.append(1/m)

        mutant = mutant*np.min(alphas)

        # # Not fool proof
        mutant = np.clip(mutant, 0, 1)

        return mutant, False

    # Perform Scaled Mutant operation
    elif handle == 'reflection':
        for i, m in enumerate(mutant):
            if m > 1:
                mutant[i] = 2-m
            elif m < 0:
                mutant[i] = -m
            else:
                mutant[i] = m

        return mutant, False
    
    else:
        raise ValueError("The constraint handle that selects how to manage boundary violations is not valid")

