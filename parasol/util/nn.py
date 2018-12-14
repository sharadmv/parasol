from deepx import T, stats

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
