import numpy as np
import numpy.linalg as l
import pandas as pd
import uuid
from typing import List, Dict
from data_structures import Human
from configs import SpatialConfig


def move(person: Human, direction_vector: np.ndarray):
    person.location_vector += direction_vector


def get_distance(person1: Human, person2: Human):
    return l.norm(person1.location_vector - person2.location_vector)


def setup(n_people):
    Human.update_forward_refs()
    people = {}
    for i in range(n_people):
        loc = np.array([np.random.uniform(config.x_lim[0], config.x_lim[1], 1)[0],
                        np.random.uniform(config.y_lim[0], config.y_lim[1], 1)[0]])

        new_person = Human(
                        id=str(uuid.uuid4()),
                        currently_at=[],
                        infected=False,
                        recovered=False,
                        susceptible=True,
                        dead=False,
                        time_since_infected=0,
                        location_vector=loc
                    )

        people[new_person.id] = new_person

    return people


def get_summary_stats_from_town(town: List[Human]):
    no_dead = [person.dead for person in town].count(True)
    no_infected = [person.infected for person in town].count(True)
    no_recovered = [person.recovered for person in town].count(True)
    no_susceptible = [person.susceptible for person in town].count(True)
    no_not_infected = [person.infected for person in town].count(False) - no_dead
    total_no = len(town)

    stats = {
        "total_no": total_no,
        "no_dead": no_dead,
        "no_infected": no_infected,
        "no_recovered": no_recovered,
        "no_susceptible": no_susceptible,
        "no_not_infected": no_not_infected
    }

    return stats


def simulate(config: SpatialConfig, people: Dict[str, Human]):
    for person in people.values():
        person.location_vector = np.random.uniform(config.x_lim[0], config.x_lim[1], 2)

    stats = {}
    for i in range(config.time):
        for person in people.values():
            move(person, np.random.uniform(-3, 3, 2))
            if not person.infected:
                continue

            for person2 in people.values():
                if person2.infected or (get_distance(person, person2) > config.infection_distance) or (
                        not person2.susceptible):
                    continue
                u = np.random.random()
                if u < config.probability_of_infection:
                    person2.infected = True
                    person2.susceptible = False
                    person2.time_since_infected = 0

        stats[i] = get_summary_stats_from_town(list(people.values()))

    return people, pd.DataFrame(stats)


if __name__ == "__main__":
    config = SpatialConfig()

    people, stats = simulate(config, setup(100))

    print(
        f'Number of infected people after {config.time} time steps: {sum([person.infected for person in people.values()])}')
    print(
        f'Number of dead people after {config.time} time steps: {sum([person.dead for person in people.values()])}')
