import six
from abc import abstractmethod, ABCMeta
from deepx import T

@six.add_metaclass(ABCMeta)
class Model(object):

  def __init__(self):
      self.session = T.interactive_session()

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
