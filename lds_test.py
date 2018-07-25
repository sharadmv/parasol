import numpy as np
np.set_printoptions(suppress=True, precision=4)
from deepx import stats, T

from lds import lds_inference

N = 1
H = 10
ds = 2
du = 2

np.random.seed(0)

A = np.tile(np.eye(ds)[None], [H, 1, 1])
# B = np.tile(np.eye(du)[None], [H, 1, 1])
B = np.zeros([H, ds, du])
Q = np.tile(np.eye(ds)[None], [H, 1, 1])
R = np.eye(ds) * 1e-6

def generate_data(N):
    X = np.zeros([H, N, ds])
    Y = np.zeros([H, N, ds])
    # U = np.random.normal(size=[H, N, du])
    U = np.zeros([H, N, du])

    for i in range(N):
        for t in range(H - 1):
            mean = (
                np.einsum('ab,b->a', A[t], X[t, i]) + np.einsum('ab,b->a', B[t], U[t, i])
            )
            cov = Q[t]
            X[t + 1, i] = np.random.multivariate_normal(mean=mean, cov=cov)
            Y[t, i] = np.random.multivariate_normal(mean=X[t, i],
                                                    cov=R)
        Y[-1, i] = np.random.multivariate_normal(mean=X[-1, i], cov=R)
    return X, Y, U

X, Y, U = generate_data(N)

p_Y = stats.Gaussian([
    T.to_float(np.tile(R[None, None], [H, N, 1, 1])),
    T.to_float(Y)
])
p_U = stats.Gaussian([
    T.eye(du, batch_shape=[H, N]),
    T.to_float(U)
])

p_X = lds_inference(p_Y, p_U, tuple(map(T.to_float, (A, B, Q))), smooth=False)
# p_X_smoothed = lds_inference(p_Y, p_U, tuple(map(T.to_float, (A, B, Q))), smooth=True)

sess = T.interactive_session()
results = sess.run(p_X.get_parameters('regular'))
# results_smoothed = sess.run(p_X_smoothed.get_parameters('regular'))
mean = results[1]
# mean_s = results_smoothed[1]
