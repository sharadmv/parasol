import numpy as np
from deepx import T, stats
from scipy.ndimage import filters

def map_network(network):
    def map_fn(data):
        data_shape = T.shape(data)
        leading = data_shape[:-1]
        dim_in = data_shape[-1]
        flattened = T.reshape(data, [-1, dim_in])
        net_out = network(flattened)
        if isinstance(net_out, stats.Gaussian):
            sigma, mu = net_out.get_parameters('regular')
            dim_out = T.shape(mu)[-1]
            return stats.Gaussian([
                T.reshape(sigma, T.concatenate([leading, [dim_out, dim_out]])),
                T.reshape(mu, T.concatenate([leading, [dim_out]])),
            ])
        elif isinstance(net_out, stats.Bernoulli):
            params = net_out.get_parameters('natural')
            dim_out = T.shape(params)[-1]
            return stats.Bernoulli(
                T.reshape(params, T.concatenate([leading, [dim_out]]))
            , 'natural')
        else:
            raise Exception("Unimplemented distribution")
    return map_fn

def chunk(*data, **kwargs):
    chunk_size = kwargs.pop('chunk_size', 100)
    shuffle = kwargs.pop('shuffle', False)
    N = len(data[0])
    if shuffle:
        permutation = np.random.permutation(N)
    else:
        permutation = np.arange(N)
    num_chunks = N // chunk_size
    if N % chunk_size > 0:
        num_chunks += 1
    for c in range(num_chunks):
        chunk_slice = slice(c * chunk_size, (c + 1) * chunk_size)
        idx = permutation[chunk_slice]
        yield idx, tuple(d[idx] for d in data)

def generate_noise(dims, std=1.0, smooth=False):
    if std == 0.0:
        return np.zeros(dims)
    noise = std * np.random.randn(*dims)
    if smooth:
        for j in range(dims[-1]):
            noise[..., j] = filters.gaussian_filter(noise[..., j], 2.0)
        emp_std = np.std(noise, axis=0)
        noise = std * (noise / emp_std)
    return noise
