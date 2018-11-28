import numpy as np
from deepx import T, stats

from .common import Dynamics

__all__ = ['NNDS']

class NNDS(Dynamics):

    def __init__(self, network):
        self.network = network
        self.cache = {}

    def forward(self, q_Xt, q_At):
        Xt, At = q_Xt.sample()[0], q_At.sample()[0]
        ds, da = T.shape(Xt)[-1], T.shape(At)[-1]
        leading_dim = T.shape(Xt)[:-1]
        Xt = T.reshape(Xt, [-1, ds])
        At = T.reshape(At, [-1, da])
        XAt = T.concatenate([Xt, At], -1)
        p_Xt1 = self.network(XAt)
        if isinstance(p_Xt1, stats.Gaussian):
            return stats.Gaussian([
                T.reshape(p_Xt1.get_parameters('regular')[0], T.concatenate([leading_dim, [ds, ds]])),
                T.reshape(p_Xt1.get_parameters('regular')[1], T.concatenate([leading_dim, [ds]])),
            ])
        else:
            raise Exception()

    def kl_gradients(self, q_Xt, q_At, _):
        kl = self.kl_divergence(q_Xt, q_At, _)
        params = self.network.get_parameters()
        return list(zip(params, T.grad(kl, params)))

    def kl_divergence(self, q_X, q_A, _):
        # q_Xt - [N, H, ds]
        # q_At - [N, H, da]
        if (q_X, q_A) not in self.cache:
            q_Xt = stats.Gaussian([
                q_X.get_parameters('regular')[0][:, :-1],
                q_X.get_parameters('regular')[1][:, :-1],
            ])
            q_At = stats.Gaussian([
                q_A.get_parameters('regular')[0][:, :-1],
                q_A.get_parameters('regular')[1][:, :-1],
            ])
            p_Xt1 = self.forward(q_Xt, q_At)
            q_Xt1 = stats.Gaussian([
                q_X.get_parameters('regular')[0][:, 1:],
                q_X.get_parameters('regular')[1][:, 1:],
            ])
            self.cache[(q_X, q_A)] = stats.kl_divergence(q_Xt1, p_Xt1)
        return self.cache[(q_X, q_A)]
