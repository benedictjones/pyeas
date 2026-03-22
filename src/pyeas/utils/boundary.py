import numpy as np 
from typing import Optional, Literal, Tuple


def count_bound_violation(mutant:np.ndarray) -> int:
    """
    Count the number of boundary violations

    Args:
        mutant (np.ndarray): normalised population member array

    Returns:
        int: number of genes below 0 or above 1
    """
    num_violations = np.size(np.where(mutant < 0)) + np.size(np.where(mutant > 1))  # How many clips are there?
    return num_violations


def handle_bound_violation(
        mutant: np.ndarray, 
        handle: Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']] = 'reflection',
) -> Tuple[np.ndarray, bool]:
    """_summary_

    Args:
        mutant (np.ndarray): 
            Normalised population member array
        handle (Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']], optional): 
            Method to handle any violations.
            Defaults to 'reflection'.

    Returns:
        Tuple[np.ndarray, bool]: adjusted population member and whether there are no violations
    """

    num_violations = count_bound_violation(mutant)

    # If no violations, just return as nothing to do
    if num_violations == 0:
        return mutant, True


    # No handling
    if handle is None:
        pass
    
    # Perform projection (i.e., clipping)
    elif handle == 'clip' or handle == 'projection':
        mutant = np.clip(mutant, 0, 1)


    # Return the resample flag
    elif handle == 'resample':
        pass 

    # Perform Scaled Mutant operation
    elif handle == 'scaled':
        alphas = [1]
        for m in mutant:
            if m > 1:
                alphas.append(1/m)

        mutant = mutant*np.min(alphas)

        # # Not fool proof
        mutant = np.clip(mutant, 0, 1)

    # Perform Scaled Mutant operation
    elif handle == 'reflection':
        for i, m in enumerate(mutant):
            if m > 1:
                mutant[i] = 2-m
            elif m < 0:
                mutant[i] = -m
            else:
                mutant[i] = m

    
    else:
        raise ValueError("The constraint handle that selects how to manage boundary violations is not valid")


    # Check again and return
    num_violations = count_bound_violation(mutant)
    return mutant, num_violations==0



def retry_boundary(
        mutant:np.ndarray, 
        handle: Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']],
    ) -> np.ndarray:
    """
    Ensures that a trial/mutant pop member does not violate given boundaries via multiple retries.

    Args:
        mutant (np.ndarray): _description_
        handle (Optional[Literal['clip', 'projection', 'resample', 'scaled', 'reflection']]): _description_

    Returns:
        np.ndarray: handled population candidate
    """
    no_violations_left = False
    resample_count = 0
    while no_violations_left is False:

        # If the mutants values violate the bounds, deal with it
        checked_mutant, no_violations_left = handle_bound_violation(mutant, handle=handle)
        resample_count += 1

        if resample_count >= 100:
            checked_mutant, no_violations_left = handle_bound_violation(mutant, handle='clip')
            break 

    return checked_mutant