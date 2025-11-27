import numpy as np
from typing import Optional, Union, Mapping, Literal, TypedDict, Tuple, List
import logging
from dataclasses import dataclass
import numbers 
import copy 

logger = logging.getLogger(__name__)


@dataclass
class Genes:
    """
    Define a group of genes which all have the same bounds.
    """
    bounds: Tuple[numbers.Real, numbers.Real]
    number: int

    def __post_init__(self):
        """
        Perform type check on the gene(s) properties.
        """
        if (
            isinstance(self.bounds, tuple) is False
            or len(self.bounds) != 2
            or isinstance(self.bounds[0], numbers.Real) is False 
            or isinstance(self.bounds[1], numbers.Real) is False 
        ):
            raise TypeError(f"Bounds must be a tuple of two floats! Not: {self.bounds}")

        if isinstance(self.number, int) is False:
            raise TypeError(f"Number of genes with the bounds (in this grouping) must be an integer! Not: {self.bounds}")
        

class Population:
    """ 
    Used to convert populations from a real valued (possibly grouped) format,
    to a normalised (flattened) format, and back.            
    """    

    @property
    def size(self) -> int:
        """Population size (i.e., number of members)"""
        return self._size
    
    @property
    def bounds(self) -> int:
        """Bounds for each gene within a member"""
        return np.array(self._bounds)
    
    @property
    def groupings(self) -> int:
        """The gene group sizes"""
        return np.array(self._groupings)
    
    @property
    def n_groups(self) -> int:
        """The gene group sizes"""
        return None if self.flatten_gene_groupings is True else len(self._groupings)
    
    @property
    def n_dimensions(self) -> int:
        """Number of genes in each member"""
        return self._n_dimensions
    
    @property
    def initi_method(self) -> int:
        return self._initi_method
    
    @property
    def population_raw(self) -> np.ndarray:
        """Get raw normalised population

        Returns:
            np.ndarray: raw pop
        """
        return self._normalised_population
    
    @population_raw.setter
    def population_raw(self, arr:np.ndarray):
        """Get raw normalised population

        Returns:
            np.ndarray: raw pop
        """
        if isinstance(arr, np.ndarray) is False:
            raise TypeError(f"Must set the population using an array, not a {type(arr)}")

        if np.shape(arr)[0] != self.size:
            raise ValueError(f"Array to set as population must have {self.size} rows, but is shape: {np.shape(arr)}")
        
        if np.shape(arr)[1] != self.n_dimensions:
            raise ValueError(f"Array to set as population must have {self.size} rows, but is shape: {np.shape(arr)}")
        
        self._normalised_population = arr
        
        return
    
    @property
    def population(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent population"""
        
        denormalised_pop = self.denormalise(self._normalised_population)
        
        if self.flatten_gene_groupings is True:
            return denormalised_pop

        return self.group(denormalised_pop)

    @population.setter
    def population(self, arr:np.ndarray):
        """ 
        Set the population to a new value.
        Several checks are made when doing so.

        Args:
            arr (np.ndarray): new population
        """
        if isinstance(arr, np.ndarray) is False:
            raise TypeError(f"Must set the population using an array, not a {type(arr)}")

        if np.shape(arr)[0] != self.size:
            raise ValueError(f"Array to set as population must have {self.size} rows, but is shape: {np.shape(arr)}")

        if self.flatten_gene_groupings is False:
            if len(np.shape(arr)) != 2:
                raise ValueError(f"Array must be {self.size} members by {self.n_groups} sub array groupings, but is shape: {np.shape(arr)}")
            
            if np.shape(arr)[1] != self.n_groups:
                raise ValueError(f"Array to set as population must have {self.n_groups} groups per row, but is shape: {np.shape(arr)}")
            
            arr = self.ungroup(arr)

        else:
            
            if np.shape(arr)[1] != self.n_dimensions:
                raise ValueError(f"Array to set as population must have {self.n_dimensions} genes per row, but is shape: {np.shape(arr)}")        

        arr = self.normalise(arr)

        if np.min(arr) < 0:
            logger.warning(f"[Population] [Setter] Normalised population to set contains values below 0! Clipping...")
            arr = np.clip(arr, min=0)
        if np.max(arr) > 1:
            logger.warning(f"[Population] [Setter] Normalised population to set contains values above 1! Clipping...")
            arr = np.clip(arr, max=1)

        self._normalised_population = arr
        logger.info(f"[Population] [Setter] Finished setting population!")


    def __repr__(self) -> str:
        """What is shown when the object is called"""
        summary = [f"Population of {self._size} members, each with {self._n_dimensions} genes. "]

        if self.flatten_gene_groupings:
            summary.append(f"Genes groupings are flattened.")    
        else:
            summary.append(f"Genes are grouped as follows: {self._groupings}.")   

        summary.append(f"Population is: \n{self.population[:5]}\n ...")

        return "\n".join(summary)
    

    def __init__(
        self,
        size:int|float,
        member:List[Genes],
        seed:Optional[int]=None,
        initi_method:Literal['uniform']='uniform',
        use_absolute_size:bool=True,
        flatten_gene_groupings:bool=True,
    ):
        """
        Initilaise new random population.

        Args:
            size (int | float): 
                Number of population members to create
            member (List[Genes]): 
                Definition of the member, using a list of Gene 
                DataClasses to define the number of genes and
                associated bounds.
            seed (Optional[int], optional): 
                Random state. 
                Defaults to None.
            initi_method (Literal["uniform"], optional): 
                Method used to create population. 
                Defaults to 'uniform'.
            use_absolute_size (bool, optional): 
                Whether to assign size using the absolute integer value (True)
                Or whether to set the population size as a multiple of the 
                number of dimensions (False).
                Defaults to True.
            flatten_gene_groupings (bool, optional): 
                Whether to flatten the population so it is a simple 2D array (True), or
                whether to retain the Gene grouping defined in the member list. 
                Defaults to True.
        """

        # # Make random generator object
        self._rng = np.random.default_rng(seed)

        if len(member) == 1 and flatten_gene_groupings is False:
            logging.warning(f"[Population] You are retaining gene groupings, but there is only one gene")


        self._initi_method = initi_method
        self.use_absolute_size = use_absolute_size
        self.flatten_gene_groupings = flatten_gene_groupings
        

        if isinstance(member, list) is False:
            raise TypeError("Must be using a list of Genes to define a member: Not using a list!")

        self._groupings = []
        self._bounds = []
        self._n_dimensions = 0
        for element in member:
            if isinstance(element, Genes) is False:
                raise TypeError("Must be using a list of Genes to define a member: not using a Gene element object!")
            
            self._groupings.append(element.number)
            self._bounds += [element.bounds]*element.number
            self._n_dimensions += element.number
        self._bounds = np.array(self._bounds)

        if flatten_gene_groupings is False:
            logger.info(f"[Population] Members contain genes grouped in groups of: {self._groupings}")

        logger.info(f"[Population] Each Member has {self._n_dimensions} genes")
        info = ', '.join([f"N={element.number} ∈{element.bounds}" for element in member])
        logger.info(f"[Population] Gene members: {info}")

        if size is not None and use_absolute_size is False:
            self._size = int(size*self._n_dimensions)
        elif size is not None and use_absolute_size is True:
            if isinstance(size, int) is False:
                raise TypeError(f"Size={size} is not an integer!! To use a float set use_absolute_size=False ")
            self._size = size
        
        logger.info(f"[Population] Population size (number of members) is {self._size}")

        
        if self._initi_method == 'uniform':
            normalised_population = self._rng.random((self._size, self._n_dimensions))

        # elif self._initi_method == 'grid':       

        else:
            raise ValueError(f"Select a valid population initialization method")

        self._normalised_population = np.around(
            normalised_population,
            decimals=6,
        )
        
        return
    
    #
    
    def copy(self):
        """
        Make a copy of the current population object.

        Returns:
            Population: deep copy of self
        """
        return copy.deepcopy(self)

    #

    def denormalise(self, population: np.ndarray) -> np.ndarray:
        """
        Produce the de-normalised population using the bounds.
        Also groups the population into its gene groups (if 'groupings' is being used).

        Args:
            population (np.ndarray): raw (normalised) population 2D array

        Returns:
            np.ndarray: population
        """

        if len(np.shape(population)) != 2:
            raise ValueError(f"Can only group a 2D array!")
        elif isinstance(population[0,0], numbers.Real) is False:
            raise ValueError(f"Population must be a simple 2D array of numbers!")
        # elif np.shape(population)[0] != self.size:
        #     raise ValueError(f"Population to group must have correct number of members")
        elif np.shape(population)[1] != self.n_dimensions:
            raise ValueError(f"Population to group must have correct number genes (i.e., the dimension)")
        
        population_denomalised = self.bounds[:,0] + population*abs(self.bounds[:,1]-self.bounds[:,0])
        population_denomalised = np.around(
            population_denomalised.astype(np.float64), 
            decimals=5,
        )
        
        return population_denomalised

    #

    def group(self, population: np.ndarray) -> np.ndarray:
        """
        Group the population members into its gene groups.

        Args:
            population (np.ndarray): Un grouped raw population

        Returns:
            np.ndarray: grouped population
        """

        if len(np.shape(population)) != 2:
            raise ValueError(f"Can only group a 2D array!")
        elif isinstance(population[0,0], numbers.Real) is False:
            raise ValueError(f"Population must be a simple 2D array of numbers!")
        elif np.shape(population)[0] != self.size:
            raise ValueError(f"Population to group must have correct number of members")
        elif np.shape(population)[1] != self.n_dimensions:
            raise ValueError(f"Population to group must have correct number genes (i.e., the dimension)")
        

        grouped_chunks = []
        st = 0
        for j, n_genes in enumerate(self._groupings):
            group = np.array(population[:, st:(st+n_genes)])
            grouped_chunks.append(group)
            st += n_genes

        grouped_population_members = []
        for member in range(self.size):
            grouped_population_members.append(
                np.array([chunk[member] for chunk in grouped_chunks], dtype=object)
            )

        return np.asarray(grouped_population_members, dtype=object) 
    


    def normalise(self, population: np.ndarray) -> np.ndarray:
        """
        Produce the normalised population using the bounds.
        Also un-groups the population (if 'groupings' is being used).

        Args:
            population (np.ndarray): real values population

        Returns:
            np.ndarray: normalised population with values within [0,1]
        """

        if len(np.shape(population)) != 2:
            raise ValueError(f"Can only group a 2D array!")
        elif isinstance(population[0,0], numbers.Real) is False:
            raise ValueError(f"Population must be a simple 2D array of numbers!")
        # elif np.shape(population)[0] != self.size:
        #     raise ValueError(f"Population to group must have correct number of members")
        elif np.shape(population)[1] != self.n_dimensions:
            raise ValueError(f"Population to group must have correct number genes (i.e., the dimension)")
        
        pop_norm = (population - self._bounds[:,0])/(self._bounds[:,1] - self._bounds[:,0])
        return np.asarray(pop_norm)

    def ungroup(self, population: np.ndarray) -> np.ndarray:
        """
        Ungroup the real valued population.

        Args:
            population (np.ndarray): real population

        Returns:
            np.ndarray: ungrouped population 2D array
        """

        ungrouped_pop = []
        for member in population:
            ungrouped_pop.append(
                np.concat(member)
            )

        return np.asarray(ungrouped_pop)

    #

    # fin
    
    