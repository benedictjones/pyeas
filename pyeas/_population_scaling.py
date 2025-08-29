import numpy as np
from typing import Optional, Union


class PopScale:
    """ 
    Used to convert populations from a real valued (possibly grouped) format,
    to a normalised (flattened) format, and back.            
    """    

    def __init__(
        self,
        bounds: np.ndarray,
        groupings: Optional[Union[np.ndarray, list]] = None,
    ):
        """_summary_

        Args:
            bounds (np.ndarray): 
                Lower and upper domain boundaries for either
                    i) each parameter,
                    ii) each grouping of paramaters (see 'groupings' argument)
            groupings (Optional[Union[np.ndarray, list]], optional): 
                An array which informs the object of the shape of a population member (optional).
                None        --> each member is a 1d array
                                e.g., possible member: [1.5, 0.5, 0.6, -0.9, 1.1]
                Otherwise   --> each member contains several diffenet shaped arrays 
                                e.g., groupings=[1,3,2] 
                                      possible member: [ [1.5], [0.5, 0.6, -0.9], [1.1]] 
                Defaults to None.
        """

        self._groupings = groupings
        self._bounds = np.array(bounds)

        return
    
    #

    def _denorm(self, norm_pop_in: np.ndarray) -> np.ndarray:
        """
        Produce the de-normalised population using the bounds.
        Also groups the population into its gene groups (if 'groupings' is being used).
        """

        if self._groupings is None:
            members_denorm = self._bounds[:,0] + norm_pop_in*abs(self._bounds[:,1]-self._bounds[:,0])
            pop_denorm = np.around(members_denorm.astype(np.float64), decimals=5)
        else:
            pop_denorm = []
            for i, member in enumerate(norm_pop_in):
                pop_denorm.append(self._group_denorm(member))
        
        return np.asarray(pop_denorm, dtype=object)

    #

    def _group_denorm(self, arr: np.ndarray) -> np.ndarray:
        """
        Group the population members into its gene groups.
        """
        grouped_arr = []
        st = 0
        for j, size in enumerate(self._groupings):
            lower_b, upper_b = self._bounds[j]  # get group bounds
            group = arr[st:(st+size)]
            group_denorm = lower_b + group*(abs(upper_b-lower_b))
            grouped_arr.append(np.around(group_denorm.astype(np.float64), decimals=5))
            st += size

        return np.asarray(grouped_arr, dtype=object) 
    
    #

    def _norm(self, pop_in: np.ndarray) -> np.ndarray:
        """
        Produce the normalised population using the bounds.
        Also un-groups the population (if 'groupings' is being used).
        """

        if self._groupings is None:
            pop_norm = (pop_in - self._bounds[:,0])/(self._bounds[:,1] - self._bounds[:,0])
        else:
            pop_norm = np.array(pop_in)
            for j, grouping in enumerate(self._groupings):
                lower_b, upper_b = self._bounds[j]  # get group bounds
                pop_norm[:,j] = (pop_norm[:,j] - lower_b)/(upper_b - lower_b)
            pop_norm = np.array([np.concatenate(x) for x in pop_norm])

        pop_norm = np.around(pop_norm.astype(np.float64), decimals=5)

        return np.asarray(pop_norm)

    #

    def _ungroup_norm(self, grouped_arr: np.ndarray) -> np.ndarray:

        norm_member = []
        for j, group in enumerate(grouped_arr):
            lower_b, upper_b = self._bounds[j]  # get group bounds
            group_norm = (group - lower_b)/(upper_b - lower_b)
            norm_member.append(np.around(group_norm.astype(np.float64), decimals=5))

        return np.concatenate(norm_member, axis=0)

    # fin
    
    