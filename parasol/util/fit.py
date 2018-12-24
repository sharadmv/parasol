import tqdm
import numpy as np
import tensorflow as tf

def linear_fit(Xy, idx_x, idx_y, reg=1e-6, prior=None):
    N = Xy.shape[0]
    mu = Xy.mean(axis=0)
    empsig = np.einsum('ia,ib->ab', Xy - mu, Xy - mu)
    sigma = 0.5 * (empsig + empsig.T) / N

    if prior:
        mu0, Phi, m, n0 = prior
        sigma = (N * sigma + Phi +
                 (N * m) / (N + m) * np.outer(mu - mu0, mu - mu0)) / (N + n0)

    sigma[idx_x, idx_x] += np.eye(idx_x.stop) * reg
    mat = np.linalg.solve(sigma[idx_x, idx_x], sigma[idx_x, idx_y]).T
    lin = mu[idx_y] - mat.dot(mu[idx_x])
    cov = sigma[idx_y, idx_y] - mat.dot(sigma[idx_x, idx_x]).dot(mat.T)
    return mat, lin, cov

global_step = 0
def quadratic_regression_pd(SA, costs, diag_cost=False):
    assert not diag_cost
    global global_step
    dsa = SA.shape[-1]
    C = tf.get_variable('cost_mat{}'.format(global_step), shape=[dsa, dsa],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    L = tf.matrix_band_part(C, 0, 0)
    # L = tf.matrix_set_diag(L, tf.maximum(tf.matrix_diag_part(L), 1e-1))
    LL = tf.matmul(L, tf.transpose(L))
    c = tf.get_variable('cost_vec{}'.format(global_step), shape=[dsa],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
    s_ = tf.placeholder(tf.float32, [None, dsa])
    c_ = tf.placeholder(tf.float32, [None])
    pred_cost = 0.5 * tf.einsum('na,ab,nb->n', s_, LL, s_) + \
            tf.einsum('na,a->n', s_, c)
    mse = tf.reduce_mean(tf.square(pred_cost - c_))
    opt = tf.train.GradientDescentOptimizer(1e-4).minimize(mse)
    N = SA.shape[0]
    SA = SA.reshape([-1, dsa])
    costs = costs.reshape([-1])
    with tf.Session() as sess:
        sess.run([C.initializer, c.initializer])
        i, perm = 0, np.random.permutation(N)
        for itr in tqdm.trange(10000, desc='Fitting cost'):
            if i + 1 > N:
                i, perm = 0, np.random.permutation(N)
            idx = perm[i]
            i += 1
            _, m = sess.run([opt, mse], feed_dict={
                s_: SA,
                c_: costs,
            })
            if itr == 0 or itr == 9999:
                print('mse itr {}: {}'.format(itr, m))
        cost_mat, cost_vec = sess.run((LL, c))

    global_step += 1
    return cost_mat, cost_vec

def quadratic_regression(SA, costs, diag_cost=False):
    N, T = SA.shape[:2]
    dsa = SA.shape[-1]
    SA = SA.reshape([-1, dsa])
    if diag_cost:
        dq, quad = dsa * 2, 0.5 * np.square(SA).reshape((N*T, dsa))
    else:
        dq = dsa ** 2 + dsa
        quad = 0.5 * np.einsum('na,nb->nab', SA, SA)
        quad = quad.reshape((N*T, dsa ** 2))
    Q, _, _ = linear_fit(
            np.concatenate([
                quad, SA, costs.reshape((N*T, 1))
            ], axis=-1),
            slice(dq), slice(dq, dq + 1),
    )
    if diag_cost:
        C = np.diag(Q[0, :dsa])
        c = Q[0, dsa:]
    else:
        C = Q[0, :dsa ** 2].reshape((dsa, dsa))
        c = Q[0, dsa ** 2:]
    return C, c
