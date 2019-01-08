import numpy as np

from ..env import ParasolEnvironment

__all__ = ['Rotation']

def shape_check(mats, shapes):
    try:
        for i in range(len(mats)):
            mat, shape = mats[i], shapes[i]
            for j in range(len(mat.shape)):
                if mat.shape[j] != shape[j]:
                    return False
        return True
    except IndexError:
        return False

# from pylds
def random_rotation(dim):
    if dim == 1:
        return np.random.rand() * np.eye(1)

    theta = np.pi / 30
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((dim, dim))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(dim, dim))[0]
    return q.dot(out).dot(q.T)


class Rotation(ParasolEnvironment):

    environment_name = 'Rotation'

    def __init__(self, dims=(2, 2, 2), horizon=100, init=None, dyn=None, obs=None, **kwargs):
        self.ds, self.do, self.da = dims
        ds, do, da = self.ds, self.do, self.da
        self.horizon = H = horizon + 1

        if init:
            assert shape_check(init, ((ds,), (ds, ds)))
            self.mu0, self.Q0 = init
        else:
            self.mu0, self.Q0 = np.ones(ds), 0.01 * np.eye(ds)

        if dyn:
            assert shape_check(dyn, ((H-1, ds, ds), (H-1, ds, da), (H-1, ds, ds)))
            self.A, self.B, self.Q = dyn
        else:
            self.A = 0.99 * np.tile(random_rotation(ds), [H-1, 1, 1])
            self.B = 0.1 * np.tile(np.random.randn(ds, da), [H-1, 1, 1])
            self.Q = 0.01 * np.tile(np.eye(ds), [H-1, 1, 1])
        print('True mat:', self.A[0])

        if obs:
            assert shape_check(obs, ((H, do, ds), (H, do, do)))
            self.C, self.R = obs
        else:
            self.C = np.tile(np.eye(ds), [H, 1, 1])
            self.R = 1e-2 * np.tile(np.eye(do), [H, 1, 1])
        super(Rotation, self).__init__(**kwargs)

    def config(self):
        return {
            'init': (self.mu0, self.Q0),
            'dyn': (self.A, self.B, self.Q),
            'obs': (self.C, self.R),
            'sliding_window': self.sliding_window
        }

    def make_summary(self, *args, **kwargs):
        pass

    def state_dim(self):
        return self.ds

    def action_dim(self):
        return self.da

    def _observe(self):
        s, _, t = self._curr_state
        mean, cov = self.C[t].dot(s), self.R[t]
        return np.random.multivariate_normal(mean, cov)

    def reset(self):
        s0 = np.random.multivariate_normal(self.mu0, self.Q0)
        self._curr_state = [s0, np.zeros(self.da), 0]
        return self.observe()

    def step(self, action):
        self._curr_state = self.dynamics(self._curr_state, action)
        done = (self._curr_state[-1] == self.horizon-1)
        return self.observe(), 0.0, done, {}

    def get_state_dim(self):
        return self.do

    def get_action_dim(self):
        return self.da

    def render(self):
        pass

    def torque_matrix(self):
        return np.eye(self.da)

    def dynamics(self, state, a):
        s, _, t = state
        mean, cov = self.A[t].dot(s) + self.B[t].dot(a), self.Q[t]
        sp = np.random.multivariate_normal(mean, cov)
        return [sp, a, t+1]

    def start_recording(self):
        pass

    def grab_frame(self):
        pass

    def stop_recording(self):
        pass
