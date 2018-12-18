import numpy as np

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
