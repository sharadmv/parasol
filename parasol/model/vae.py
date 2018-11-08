from deepx import T

from .common import Model

class VAE(Model):

  def __init__(self, observation_dim, latent_dim):
      self.encoder, self.decoder = encoder, decoder
