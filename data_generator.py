from typing import Union, Tuple
from data_producer import AbstractProducer
import numpy as np


class DataGenerator:
    """
        Class that generates data using data producer
    """
    def __init__(self, n: Union[int, Tuple[int, int]], producer: AbstractProducer):
        self.size = n
        self.producer = producer

    def generate(self): ...

    def get(self) -> np.array:
        out_array = np.array(self.size)

        return out_array
