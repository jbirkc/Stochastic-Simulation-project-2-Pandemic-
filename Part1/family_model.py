import numpy as np
from dataclasses import dataclass
import uuid
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from family_model_config import FamilyModelConfig


config = FamilyModelConfig()


@dataclass
class Human:
    id: str
    infected: bool
    recovered: bool
    susceptible: bool
    dead: bool
    has_been_infected: bool
    time_since_infected: int
    child: bool
    home_idx: int
    work_idx: int
    currently_at: List['Place']
    probability_of_death: float = config.probability_of_death
    probability_of_cured: float = config.probability_of_cured
    time_since_cured: int = 0


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


def get_prob_matrix():
    q = np.array([[0.4, 0.5, 0.1],
                  [0.5, 0.4, 0.1],
                  [0.9, 0.05, 0.05]])

    return q


def setup_town(n_families, n_workplaces=2, n_supermarkets=1):
    workplaces = [Workplace(i, {}) for i in range(n_workplaces)]
    homes = [Home(i, {}) for i in range(n_families)]
    supermarkets = [Supermarket(i, {}) for i in range(n_supermarkets)]
    for family in range(n_families):
        family_size = np.random.randint(1, 6)
        for i in range(family_size):
            u_child = np.random.random()
            u_infected = np.random.random()
            is_child = u_child < config.probability_of_child
            person = Human(
                id=str(uuid.uuid4()),
                infected=u_infected < config.probability_of_start_infection,
                recovered=False,
                susceptible=True,
                dead=False,
                time_since_infected=0,
                child=is_child,
                home_idx=family,
                work_idx=0 if is_child else np.random.choice(len(workplaces)),
                currently_at=homes,
                has_been_infected=False,
            )

            homes[family].people[person.id] = person

    town = Town(
        homes=homes,
        workplaces=workplaces,
        supermarkets=supermarkets,
        people=[person for home in homes for person in home.people.values()],
    )

    return town


def infect_person(person: Human):
    person.infected = True
    person.susceptible = False
    person.time_since_infected = 0
    person.has_been_infected = True


def get_infection_probability(place: Place):
    no_infected = 0
    for human in place.people.values():
        if human.infected:
            no_infected += 1

    return min((no_infected / len(place))*0.3, 1)


def check_death(person: Human, place: Place):
    if person.infected:
        u = np.random.random()
        if u < person.probability_of_death:
            del place.people[person.id]
            person.dead = True
            person.infected = False
            person.susceptible = False
            person.recovered = False
            return person.id
    return None


def get_transistion_probability(person: Human, q:np.ndarray):
    if isinstance(person.currently_at, Home):
        return q[0]
    elif isinstance(person.currently_at, Workplace):
        return q[1]
    elif isinstance(person.currently_at, Supermarket):
        return q[2]


def get_current_location(person):
    if isinstance(person.currently_at[0], Home):
        return person.currently_at[person.home_idx]
    elif isinstance(person.currently_at[0], Workplace):
        return person.currently_at[person.work_idx]
    elif isinstance(person.currently_at[0], Supermarket):
        return person.currently_at[0]


def move_from_place_to_place(person: Human, place_list_to: List[Place]):
    if isinstance(place_list_to[0], Workplace):
        place_list_to[person.work_idx].people[person.id] = person
    elif isinstance(place_list_to[0], Home):
        place_list_to[person.home_idx].people[person.id] = person
    elif isinstance(place_list_to[0], Supermarket):
        place_list_to[0].people[person.id] = person

    del get_current_location(person).people[person.id]
    person.currently_at = place_list_to


def get_summary_stats_from_town(town: Town):
    no_dead = [person.dead for person in town.people].count(True)
    no_infected = [person.infected for person in town.people].count(True)
    no_recovered = [person.recovered for person in town.people].count(True)
    no_susceptible = [person.susceptible for person in town.people].count(True)
    no_not_infected = [person.infected for person in town.people].count(False) - no_dead
    total_no = len(town.people)

    stats = {
        "total_no": total_no,
        "no_dead": no_dead,
        "no_infected": no_infected,
        "no_recovered": no_recovered,
        "no_susceptible": no_susceptible,
        "no_not_infected": no_not_infected
    }

    return stats


def simulate(n_time_steps, n_families, n_workplaces=2, n_supermarkets=1):
    time = n_time_steps
    town = setup_town(n_families, n_workplaces, n_supermarkets)
    q = get_prob_matrix()

    stats = {}

    for t in range(time):
        stats[t] = get_summary_stats_from_town(town)

        for human in town.people:
            if human.dead:
                continue
            current_loc = get_current_location(human)
            if human.infected:
                human.time_since_infected += 1
                u = np.random.random()
                if u < human.probability_of_cured:
                    human.infected = False
                    human.recovered = True
                    human.susceptible = False
                    human.time_since_infected = 0
                else:
                    human.time_since_infected += 1
                    human_id = check_death(human, current_loc)
                    continue
            else:
                if human.susceptible:
                    infect_p = get_infection_probability(current_loc)
                    u = np.random.random()
                else:
                    u=0
                if u < infect_p:
                    infect_person(human)
                else:
                    human.time_since_infected += 1
                    if human.time_since_infected > config.time_to_susceptible:
                        human.susceptible = True

            trans_p = get_transistion_probability(human, q)
            go_to_loc = [town.homes, town.workplaces, town.supermarkets]
            go_to_idx = np.random.choice([0, 1, 2], p=trans_p)
            go_to = go_to_loc[go_to_idx]
            if not isinstance(human.currently_at[0], type(go_to[0])):
                move_from_place_to_place(human, go_to)

    stats[t + 1] = get_summary_stats_from_town(town)
    return pd.DataFrame(stats).T


if __name__ == "__main__":
    simulations_kwargs = {
        "n_time_steps": config.time,
        "n_families": config.no_families,
        "n_workplaces": config.no_workplaces,
        "n_supermarkets": config.no_supermarkets
    }

    stat_tensor = np.zeros((config.n_simulations, config.time + 1, 6))
    for i in tqdm(range(config.n_simulations)):
        stats = simulate(**simulations_kwargs).values
        stat_tensor[i] = stats

    mean_stats = stat_tensor.mean(axis=0)
    std_stats = stat_tensor.std(axis=0)

    plt.plot(mean_stats[:, 1], linestyle="dashed", label="Dead")
    plt.fill_between([i for i in range(len(mean_stats[:, 1]))],(mean_stats[:, 1]-1.96*std_stats[:,1]), (mean_stats[:, 1]+1.96*std_stats[:,1]), alpha=.1)
    plt.plot(mean_stats[:, 2], linestyle="dashed", label="Infected")
    plt.fill_between([i for i in range(len(mean_stats[:, 2]))],(mean_stats[:, 2]-1.96*std_stats[:,2]), (mean_stats[:, 2]+1.96*std_stats[:,2]), alpha=.1)
    plt.plot(mean_stats[:, 3], linestyle="dashed", label="Recovered")
    plt.fill_between([i for i in range(len(mean_stats[:, 3]))],(mean_stats[:, 3]-1.96*std_stats[:,3]), (mean_stats[:, 3]+1.96*std_stats[:,3]), alpha=.1)
    plt.plot(mean_stats[:, 4], linestyle="dashed", label="Susceptible")
    plt.fill_between([i for i in range(len(mean_stats[:, 4]))],(mean_stats[:, 4]-1.96*std_stats[:,4]), (mean_stats[:, 4]+1.96*std_stats[:,4]), alpha=.1)
    plt.plot(mean_stats[:, 5], linestyle="dashed", label="Not Infected")
    plt.fill_between([i for i in range(len(mean_stats[:, 5]))],(mean_stats[:, 5]-1.96*std_stats[:,5]), (mean_stats[:, 5]+1.96*std_stats[:,5]), alpha=.1)
    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend()
    plt.show()
