from abc import abstractmethod, ABCMeta

class Controller(object, metaclass=ABCMeta):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def act(self, observations, actions):
        pass

    @abstractmethod
    def train(self, rollouts, train_step):
        pass
