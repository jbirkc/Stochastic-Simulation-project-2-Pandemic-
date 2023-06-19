from pydantic import BaseModel
from enum import Enum
from typing import Tuple


class LockdownType(Enum):
    NONE = 0
    ISOLATION = 1
    FULL_ISOLATION = 2
    QUARANTINE = 3


class ModelType(Enum):
    PROBABILISTIC = 0
    SIMULATION = 1


class FamilyModelConfig(BaseModel):
    n_simulations: int = 100
    no_families: int = 25
    time: int = 100
    no_workplaces: int = 6
    no_supermarkets: int = 1
    probability_of_death: float = 0.005
    probability_of_cured: float = 0.1
    time_to_susceptible: int = 20
    probability_of_child: float = 0.2
    probability_of_start_infection: float = 0.1
    incubation_time = 7
    lockdown_type: LockdownType = LockdownType.NONE
    model_type: ModelType = ModelType.PROBABILISTIC
    variable_death_rate: bool = False


class SpatialConfig(BaseModel):
    x_lim: Tuple[int] = (0, 50)
    y_lim: Tuple[int] = (0, 50)
    probability_of_infection: float = 0.8
    infection_distance: float = 4.0
    time: int = 8
