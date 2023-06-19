import warnings

import numpy as np
import uuid
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from Part1.configs import FamilyModelConfig, SpatialConfig, ModelType, LockdownType
from Part1.spatial_model import simulate as simulate_spatial
from Part1.data_structures import Human, Home, Workplace, Supermarket, Town, Place, family_config, spatial_config
from itertools import product

family_config, spatial_config = FamilyModelConfig(), SpatialConfig()


def get_prob_matrix() -> np.ndarray:
    q = np.array([[0.5, 0.3, 0.2],
                  [0.4, 0.4, 0.2],
                  [0.85, 0.1, 0.05]])

    return q


def get_isolated_prob_matrix() -> np.ndarray:
    q = np.array([[1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0]])

    return q


def setup_town(n_families, n_workplaces=2, n_supermarkets=1) -> Town:
    Human.update_forward_refs()
    workplaces = [Workplace(i, {}) for i in range(n_workplaces)]
    homes = [Home(i, {}) for i in range(n_families)]
    supermarkets = [Supermarket(i, {}) for i in range(n_supermarkets)]
    for family in range(n_families):
        family_size = np.random.randint(1, 6)
        for i in range(family_size):
            u_child = np.random.random()
            u_infected = np.random.random()
            is_child = u_child < family_config.probability_of_child

            person = Human(
                id=str(uuid.uuid4()),
                currently_at=Home,
                infected=u_infected < family_config.probability_of_start_infection,
                recovered=False,
                susceptible=True,
                dead=False,
                time_since_infected=0,
                child=is_child,
                home_idx=family,
                work_idx=0 if is_child else np.random.choice(len(workplaces)),
                location_vector=np.random.random(2),
            )

            homes[family].people[person.id] = person

    town = Town(
        homes=homes,
        workplaces=workplaces,
        supermarkets=supermarkets,
        people=[person for home in homes for person in home.people.values()],
    )

    return town


def _infect_person(person: Human) -> None:
    person.infected = True
    person.susceptible = False
    person.times_sick += 1
    person.time_since_infected = 0


def get_infection_probability(place: Place) -> float:
    no_infected = 0
    for human in place.people.values():
        if family_config.lockdown_type == LockdownType.FULL_ISOLATION:
            if human.quarantined and (human.currently_at == Home):
                continue
        if human.infected:
            no_infected += 1

    return min(no_infected / len(place), 1)


def kill_human_with_probability(person: Human, place: Place) -> bool:
    if person.infected:
        u = np.random.random()

        prob_of_death = person.probability_of_death
        if family_config.variable_death_rate:
            prob_of_death *= person.time_since_infected

        if u < prob_of_death:
            del place.people[person.id]
            person.dead = True
            person.infected = False
            person.susceptible = False
            person.recovered = False
            return True
    return False


def lockdown_family(person: Human, town: Town) -> None:
    family_infected = False
    for human in town.people:
        if human.home_idx == person.home_idx:
            if human.quarantined:
                family_infected = True
                break

    if family_infected:
        for human in town.people:
            if human.home_idx == person.home_idx:
                human.quarantined = True


def cure_human_with_probability(person: Human) -> None:
    u = np.random.random()
    if u < person.probability_of_cured:
        person.recovered = True
        person.infected = False
        person.susceptible = False
        person.time_since_cured = 0


def check_susceptible(person: Human) -> None:
    if (not person.susceptible) & (not person.infected):
        if person.time_since_infected > family_config.time_to_susceptible:
            person.susceptible = True
            person.time_since_infected = 0


def infect_person_with_probability(person: Human, current_loc: Place) -> None:
    if person.susceptible:
        infect_p = get_infection_probability(current_loc)
        u = np.random.random()
        if u < infect_p:
            _infect_person(person)


def get_transition_probability(person: Human, current_location: Place) -> np.ndarray:
    q = get_prob_matrix()
    if family_config.lockdown_type.value > 0:
        q = get_isolated_prob_matrix() if person.quarantined else get_prob_matrix()

    if isinstance(current_location, Home):
        return q[0]
    elif isinstance(current_location, Workplace):
        return q[1]
    elif isinstance(current_location, Supermarket):
        return q[2]


def get_current_location(person: Human, town: Town) -> Place:
    if person.currently_at == Home:
        return town.homes[person.home_idx]
    elif person.currently_at == Workplace:
        return town.workplaces[person.work_idx]
    elif person.currently_at == Supermarket:
        return town.supermarkets[0]


def move_from_place_to_place(person: Human, currently_at: Place, place_list_to: List[Place]) -> None:
    if isinstance(place_list_to[0], Workplace):
        place_list_to[person.work_idx].people[person.id] = person
        person.currently_at = Workplace
    elif isinstance(place_list_to[0], Home):
        place_list_to[person.home_idx].people[person.id] = person
        person.currently_at = Home
    elif isinstance(place_list_to[0], Supermarket):
        place_list_to[0].people[person.id] = person
        person.currently_at = Supermarket

    del currently_at.people[person.id]


def get_summary_stats_from_town(town: Town) -> Dict[str, int]:
    no_dead = [person.dead for person in town.people].count(True)
    no_infected = [person.infected for person in town.people].count(True)
    no_recovered = [person.recovered for person in town.people].count(True)
    no_susceptible = [person.susceptible for person in town.people].count(True)
    average_infection = 0
    if family_config.model_type == ModelType.SIMULATION:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_infection = np.nanmean([
                person.people_infected / person.times_sick if person.times_sick > 0 else np.nan for person in
                town.people
            ])
    total_no = len(town.people)

    stats = {
        "total_no": total_no,
        "no_dead": no_dead,
        "no_infected": no_infected,
        "no_recovered": no_recovered,
        "no_susceptible": no_susceptible,
        "average_infection": average_infection
    }

    return stats


def check_know_infection(human) -> None:
    if human.time_since_infected > family_config.incubation_time:
        human.quarantined = True


def move(human: Human, current_loc: Place, town: Town) -> None:
    trans_p = get_transition_probability(human, current_loc)

    go_to_loc = [town.homes, town.workplaces, town.supermarkets]
    go_to_idx = np.random.choice([0, 1, 2], p=trans_p)
    go_to = go_to_loc[go_to_idx]

    if not human.currently_at == type(go_to[0]):
        move_from_place_to_place(human, current_loc, go_to)


def simulate(n_time_steps, n_families, burn_in, n_workplaces=2, n_supermarkets=1) -> pd.DataFrame:
    time = n_time_steps

    town = setup_town(n_families, n_workplaces, n_supermarkets)
    summary_statistics = {}

    for t in range(time + burn_in):
        if t >= burn_in:
            summary_statistics[t - burn_in] = get_summary_stats_from_town(town)
        for human in town.people:
            if human.dead:
                continue
            current_loc = get_current_location(human, town)

            if t < burn_in:
                move(human, current_loc, town)
                continue

            human.time_since_infected += 1
            if human.infected:
                check_know_infection(human)
                cure_human_with_probability(human)
                is_dead = kill_human_with_probability(human, current_loc)
                if family_config.lockdown_type == LockdownType.QUARANTINE:
                    lockdown_family(human, town)
                if is_dead:
                    continue
            else:
                check_susceptible(human)
                if family_config.model_type == ModelType.PROBABILISTIC:
                    infect_person_with_probability(human, current_loc)

            move(human, current_loc, town)

        if family_config.model_type == ModelType.SIMULATION:
            for home in town.homes:
                simulate_spatial(spatial_config, home.people)
            for workplace in town.workplaces:
                simulate_spatial(spatial_config, workplace.people)
            for supermarket in town.supermarkets:
                simulate_spatial(spatial_config, supermarket.people)

    summary_statistics[n_time_steps] = get_summary_stats_from_town(town)
    return pd.DataFrame(summary_statistics).T


if __name__ == "__main__":
    family_config.variable_death_rate = True

    for model, lockdown_type in product(ModelType, LockdownType):
        family_config.model_type = model
        family_config.lockdown_type = lockdown_type
        plot_title = f"Model type: {str(family_config.model_type).split('.')[-1].lower()}, Lockdown type: {str(family_config.lockdown_type).split('.')[-1].lower()}"
        print(f"Running {family_config.n_simulations} simulations")
        print(f"with model type {family_config.model_type}")
        print(f"with lockdown type {family_config.lockdown_type}")

        simulations_kwargs = {
            "n_time_steps": family_config.time,
            "n_families": family_config.no_families,
            "burn_in": family_config.burn_in_period,
            "n_workplaces": family_config.no_workplaces,
            "n_supermarkets": family_config.no_supermarkets,
        }

        stat_tensor = np.zeros((family_config.n_simulations, family_config.time + 1, 6))
        for i in tqdm(range(family_config.n_simulations)):
            stats = simulate(**simulations_kwargs).values
            stat_tensor[i] = stats

        mean_stats = np.nanmean(stat_tensor, axis=0)
        std_stats = np.nanquantile(stat_tensor, q=[0.025, 0.975], axis=0)

        plt.figure(figsize=(10, 5))
        plt.plot(mean_stats[:, 1], linestyle="dashed")
        plt.fill_between([i for i in range(len(mean_stats[:, 1]))], std_stats[0, :, 1], std_stats[1, :, 1], alpha=.1)
        plt.plot(mean_stats[:, 2], linestyle="dashed")
        plt.fill_between([i for i in range(len(mean_stats[:, 2]))], std_stats[0, :, 2], std_stats[1, :, 2], alpha=.1)
        plt.plot(mean_stats[:, 3], linestyle="dashed")
        plt.fill_between([i for i in range(len(mean_stats[:, 3]))], std_stats[0, :, 3], std_stats[1, :, 3], alpha=.1)
        plt.plot(mean_stats[:, 4], linestyle="dashed")
        plt.fill_between([i for i in range(len(mean_stats[:, 4]))], std_stats[0, :, 4], std_stats[1, :, 4], alpha=.1)
        # Create legend for mean lines
        plt.plot([], [], linestyle="dashed", label="Dead", color="blue")
        plt.plot([], [], linestyle="dashed", label="Infected", color="orange")
        plt.plot([], [], linestyle="dashed", label="Recovered", color="green")
        plt.plot([], [], linestyle="dashed", label="Susceptible", color="red")
        plt.fill_between([], [], [], alpha=.1, label="CI (2.5-97.5%)")
        plt.legend(bbox_to_anchor=(1.3, 1),
                   fancybox=True,
                   shadow=True)
        plt.xlabel("Time")
        plt.ylabel("Number of people")
        plt.title(plot_title)
        plt.subplots_adjust(right=0.75)
        plt.show()

        if family_config.model_type == ModelType.SIMULATION:
            plt.plot(mean_stats[:, 5], linestyle="dashed", label="Average infection")
            plt.fill_between([i for i in range(len(mean_stats[:, 5]))], std_stats[0, :, 5], std_stats[1, :, 5],
                             alpha=.1)
            plt.show()
