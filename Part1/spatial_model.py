import numpy as np
import numpy.linalg as l
import pandas as pd
from pydantic import BaseModel, root_validator
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid
import pandas as pd


class Config(BaseModel):
    incubation_time: int = 50
    probability_of_infection: float = 0.8
    probability_of_death: float = 0.05
    x_lim: tuple[int] = (0, 50)
    y_lim: tuple[int] = (0, 50)
    infection_distance: float = 4.0
    time_to_cured: int = 10
    no_people: int = 100
    time: int = 100
    time_to_susceptible: int = 3
    no_simulations: int = 200


config = Config()


class Person(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    id: str
    location_vector: np.ndarray
    infected: bool
    time_since_infected: int
    incubation_time: int = config.incubation_time
    is_dead: bool = False
    know_infected: bool = False
    recovered: bool = False
    susceptible: bool = True
    cured: bool = False

    @root_validator
    def check_location_vector(cls, values):
        values['location_vector'][0] = max(min(values['location_vector'][0], config.x_lim[1]), config.x_lim[0])
        values['location_vector'][1] = max(min(values['location_vector'][1], config.y_lim[1]), config.y_lim[0])

        return values


def move(person: Person, direction_vector: np.ndarray):
    person.location_vector += direction_vector


def get_distance(person1: Person, person2: Person):
    return l.norm(person1.location_vector - person2.location_vector)


def setup(n_people):
    people = {}
    for i in range(n_people):
        loc = np.array([np.random.uniform(config.x_lim[0], config.x_lim[1], 1)[0],
                        np.random.uniform(config.y_lim[0], config.y_lim[1], 1)[0]])

        new_person = Person(id=str(uuid.uuid4()),
                            location_vector=loc,
                            infected=False,
                            time_since_infected=0)

        people[new_person.id] = new_person

    return people


def get_summary_stats_from_town(town: list[Person]):
    no_dead = [person.is_dead for person in town].count(True)
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


def simulate():
    people = setup(config.no_people)
    chosen_idx = np.random.choice(a=len(people.keys()), size=10)
    for i in chosen_idx:
        people[list(people.keys())[i]].infected = True
    stats = {}
    for i in range(config.time):
        for person in people.values():
            move(person, np.random.uniform(-1, 1, 2))
            if not person.infected:
                continue

            if person.is_dead:
                continue

            person.time_since_infected += 1
            if person.time_since_infected > config.incubation_time:
                person.know_infected = True
                continue

            if person.know_infected:
                if person.time_since_infected > config.time_to_cured:
                    person.infected = False
                    person.know_infected = False
                    person.recovered = True

            if person.cured:
                if person.time_since_infected > config.time_to_susceptible:
                    person.susceptible = True
                    person.time_since_infected = 0

            u_death = np.random.random()
            if u_death < config.probability_of_death:
                person.is_dead = True
                person.infected = False
                person.susceptible = False
                person.know_infected = False
                person.time_since_infected = 0
                continue

            for person2 in people.values():
                if person2.infected or (get_distance(person, person2) > config.infection_distance) or (
                        not person2.susceptible):
                    continue
                u = np.random.random()
                if u < config.probability_of_infection:
                    person2.infected = True
                    person2.susceptible = False
                    person2.know_infected = False

                    person2.time_since_infected = 0

        stats[i] = get_summary_stats_from_town(list(people.values()))

        plot = True
        if plot:
            for person in people.values():
                if person.is_dead:
                    color = 'k'
                elif person.know_infected:
                    color = 'y'
                elif person.infected:
                    color = 'r'
                else:
                    color = 'b'

                plt.scatter(person.location_vector[0], person.location_vector[1], c=color)
            plt.savefig(f"Part1/plots/spatial_plot/plot{i}.png")
            plt.close()

    return people, pd.DataFrame(stats)


if __name__ == "__main__":
    no_simulations = 1
    stat_tensor = np.zeros((config.no_simulations, config.time, 6))
    for i in tqdm(range(no_simulations)):
        people, stats = simulate()
        stat_tensor[i] = stats.T.values

    print(
        f'Number of infected people after {config.time} time steps: {sum([person.infected for person in people.values()])}')
    print(
        f'Number of dead people after {config.time} time steps: {sum([person.is_dead for person in people.values()])}')


    mean_stats = stat_tensor.mean(axis=0)
    std_stats = stat_tensor.std(axis=0)

    plt.plot(mean_stats[:, 1], linestyle="dashed", label="Dead")
    plt.fill_between([i for i in range(len(mean_stats[:, 1]))], (mean_stats[:, 1] - 1.96 * std_stats[:, 1]),
                     (mean_stats[:, 1] + 1.96 * std_stats[:, 1]), alpha=.1)
    plt.plot(mean_stats[:, 2], linestyle="dashed", label="Infected")
    plt.fill_between([i for i in range(len(mean_stats[:, 2]))], (mean_stats[:, 2] - 1.96 * std_stats[:, 2]),
                     (mean_stats[:, 2] + 1.96 * std_stats[:, 2]), alpha=.1)
    plt.plot(mean_stats[:, 3], linestyle="dashed", label="Recovered")
    plt.fill_between([i for i in range(len(mean_stats[:, 3]))], (mean_stats[:, 3] - 1.96 * std_stats[:, 3]),
                     (mean_stats[:, 3] + 1.96 * std_stats[:, 3]), alpha=.1)
    plt.plot(mean_stats[:, 4], linestyle="dashed", label="Susceptible")
    plt.fill_between([i for i in range(len(mean_stats[:, 4]))], (mean_stats[:, 4] - 1.96 * std_stats[:, 4]),
                     (mean_stats[:, 4] + 1.96 * std_stats[:, 4]), alpha=.1)
    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend()
    plt.show()

    t = pd.DataFrame(stats)
