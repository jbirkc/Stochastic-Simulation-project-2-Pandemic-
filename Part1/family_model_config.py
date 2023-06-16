from dataclasses import dataclass


@dataclass
class FamilyModelConfig:
    n_simulations: int = 100
    no_families: int = 10
    time: int = 100
    no_workplaces: int = 6
    no_supermarkets: int = 1
    probability_of_death: float = 0.1
    probability_of_cured: float = 0.2
    time_to_susceptible: int = 14
    probability_of_child: float = 0.2
    probability_of_start_infection: float = 0.5
