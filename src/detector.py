import abc


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame):
        pass
