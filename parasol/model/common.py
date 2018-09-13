from abc import abstractmethod, ABCMeta

class Model(object, metaclass=ABCMeta):

  def __init__(self):
    pass

  @abstractmethod
  def initialize(self, initial_model=None):
    pass

  @abstractmethod
  def train(self, data):
    pass

  @abstractmethod
  def encode(self, y, a):
    pass

  @abstractmethod
  def decode(self, x):
    pass

  @abstractmethod
  def get_dynamics(self):
    pass

  @property
  @abstractmethod
  def has_dynamics(self):
    pass
