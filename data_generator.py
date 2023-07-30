from typing import Union, Tuple
import numpy as np
from tqdm import tqdm

from data_producer import AbstractProducer


class DataGenerator:
    """
        Class that generates data using data producer
    """
    def __init__(self, n: Union[int, Tuple[int, int]], producer: AbstractProducer):
        self.size = n
        self.producer = producer

    def generate(self, x: Union[float, np.array]):
        result = np.zeros(shape=(self.size, self.producer.get_state_shape()[1]))
        for i, t in tqdm(enumerate(x)):
            result[i] = self.producer(t)
        return result
