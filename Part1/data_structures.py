from dataclasses import dataclass
from typing import List, Dict, Type

import numpy as np
from pydantic import BaseModel, root_validator

from Part1.configs import FamilyModelConfig, SpatialConfig

family_config = FamilyModelConfig()
spatial_config = SpatialConfig()


class Human(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    id: str
    infected: bool
    recovered: bool
    susceptible: bool
    dead: bool
    time_since_infected: int
    child: bool = False
    home_idx: int = None
    work_idx: int = None
    currently_at: Type['Place']
    location_vector: np.ndarray
    probability_of_death: float = family_config.probability_of_death
    probability_of_cured: float = family_config.probability_of_cured
    time_since_cured: int = 0
    quarantined: bool = False
    times_sick: int = 0
    people_infected: int = 0

    @root_validator
    def check_location_vector(cls, values):
        values['location_vector'][0] = max(min(values['location_vector'][0], spatial_config.x_lim[1]),
                                           spatial_config.x_lim[0])
        values['location_vector'][1] = max(min(values['location_vector'][1], spatial_config.y_lim[1]),
                                           spatial_config.y_lim[0])

        return values


@dataclass
class Place:
    id: int
    people: Dict[str, Human]

    def __getitem__(self, item):
        return self.people[item]

    def __len__(self):
        return len(self.people)


@dataclass
class Workplace(Place):
    pass


@dataclass
class Home(Place):
    pass


@dataclass
class Supermarket(Place):
    pass


@dataclass
class Town:
    homes: List[Home]
    workplaces: List[Workplace]
    supermarkets: List[Supermarket]
    people: List[Human]

