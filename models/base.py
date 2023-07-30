from abc import ABC, abstractmethod


class SpikedNN(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs): ...
    def predict(self, *args, **kwargs): ...
