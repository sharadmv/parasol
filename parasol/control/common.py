from abc import abstractmethod, ABCMeta

class Policy(object, metaclass=ABCMeta):

  def __init__(self):
    self.parameters = {}

  def get_parameters(self):
    return self.parameters

  def get_parameter(self, name):
    return self.parameters[name]

  def set_parameter(self, name, value):
    self.parameters[name] = value

  @abstractmethod
  def initialize(self, initial_policy=None):
    pass

  @abstractmethod
  def act(self, observations, actions):
    pass

  @abstractmethod
  def train(self, data):
    pass

  def preprocess(self):
    pass

  def postprocess(self):
    pass
