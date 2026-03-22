import numpy as np
from typing import Optional, Union, Mapping, Literal, TypedDict, Tuple, List, Any
import logging
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator


logger = logging.getLogger(__name__)


def ensure_nd_array(arr:Any, n_dim:int):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Trial must be an array")
    if arr.ndim != n_dim:
        raise ValueError(f"Trial must be a {n_dim} d population")


class HistoryEvals(BaseModel):
    """Data class to hold trial history"""
    model_config = ConfigDict(validate_assignment=True)
    n_evals: List[int] = Field([], description="History of number of population member evaluations")
    
    def append_evals(self, n_evals:np.ndarray):
        self.n_evals = self.n_evals + [n_evals]

    @field_validator('n_evals', mode='after')
    @classmethod
    def ensure_n_evals(cls, value: List[Any]) -> List[Any]:
        if len(value) > 0:
            for v in value:
                if not isinstance(v, int):
                    raise ValueError(f"Number evaluations must be an int")
        return value
    
class HistoryTrials(BaseModel):
    """Data class to hold trial history"""
    
    # Allow arbitrary types so Pydantic doesn't complain about the np.ndarray class
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    trials: List[np.ndarray] = Field([], description="History of trial populations")
    
    def append_trial(self, trial:np.ndarray):
        self.trials = self.trials + [trial]

    def get_trial_means(self) -> List[np.ndarray]:            
        return [np.mean(trial, axis=0) for trial in self.trials]

    def get_trial_std(self) -> List[np.ndarray]:            
        return [np.std(trial, axis=0) for trial in self.trials]
    
    @field_validator('trials', mode='after')
    @classmethod
    def ensure_trials(cls, value: Any) -> any:
        # Only validate if we have trial populations stored
        if len(value) > 0:
            # Check each trial pop
            for arr in value:
                ensure_nd_array(arr, 2)
        return value


class Solution(BaseModel):
    """Data class to hold trial history"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    member: np.ndarray = Field(description="A population member")
    loss: float = Field(description="Members corresponding recorded loss")

class HistoryBestParent(BaseModel):
    """Data class to hold trial history"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    parents: List[np.ndarray] = Field([], description="List of best parent members")
    parent_losses: List[float] = Field([], description="List of best parent losses")

    
    def append_parent(self, member:np.ndarray, loss:float):
        self.parents = self.parents + [member]
        self.parent_losses = self.parent_losses + [loss]

    @computed_field
    @property
    def generation(self) -> int:
        """Get the current generation."""
        return len(self.parents)
    
    @computed_field
    @property
    def best(self) -> Optional[Solution]:
        """Fetch the best parent solution."""
        if len(self.parent_losses) <= 0:
            return None
        idx = np.argmin(self.parent_losses)
        return Solution(
            member=self.parents[idx],
            loss=self.parent_losses[idx],
        )

    @computed_field
    @property
    def parent_best(self) -> Optional[Solution]:
        """Fetch the best parent solution."""
        return Solution(
            member=self.parents[-1],
            loss=self.parent_losses[-1],
        )
    
    @field_validator('parents', mode='after')
    @classmethod
    def ensure_parents(cls, value: List[Any]) -> List[Any]:
        if len(value) > 0:
            for arr in value:
                ensure_nd_array(arr, 1)
        return value

    @field_validator('parent_losses', mode='after')
    @classmethod
    def ensure_parent_losses(cls, value: List[Any]) -> List[Any]:
        if len(value) > 0:
            for v in value:
                if not isinstance(v, float):
                    raise ValueError(f"Parent losses must be a float")
        return value