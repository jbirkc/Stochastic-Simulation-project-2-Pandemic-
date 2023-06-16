import numpy as np
import numpy.linalg as l
from dataclasses import dataclass


@dataclass
class Person:
    location_vector: np.ndarray
    infected: bool


