from deepx import T, stats

def lds_inference(p_X, p_U, dynamics, smooth=False):

    A, B, Q = dynamics
    D = T.shape(Q)[-1]

    J_22 = T.matrix_inverse(Q)
    J_12 = -T.einsum('tba,tbc->tac', A, J_22)
    J_21 = T.matrix_transpose(J_12)
    J_11 = -T.einsum('tab,tbc->tac', J_12, A)

    U = p_U.expected_value()
    h2 = T.einsum('tia,tba,tbc->tic', U, B, J_22)
    h1 = -T.einsum('tia,tab->tib', h2, A)
    sess = T.interactive_session()
    import ipdb; ipdb.set_trace()

    potentials = (
        T.matrix_inverse(p_X.get_parameters('regular')[0]),
        T.matrix_solve(p_X.get_parameters('regular')[0],
                       p_X.get_parameters('regular')[1][..., None])[..., 0],
    )
    batch_size = T.shape(potentials[1])[1]

    def kalman_filter(previous, potential):
        t, prev, _ = previous
        J_tt = prev[0] + potential[0]
        h_tt = prev[1] + potential[1]
        mat_inv = T.einsum('ab,ibc->iac', J_21[t], T.matrix_inverse(J_tt + J_11[t][None]))
        J_t1_t = J_22[t][None] - T.einsum('iab,bc->iac', mat_inv, J_12[t])
        h_t1_t = h2[t] - T.einsum('iab,ib->ib', mat_inv, h_tt + h1[t])

        return t + 1, (J_tt, h_tt), (J_t1_t, h_t1_t)

    _, filtered, _ = T.scan(kalman_filter, potentials,
                       (0,
                        (T.eye(D, batch_shape=[batch_size]) * 0.001,
                         T.zeros([batch_size, D])),
                        (T.eye(D, batch_shape=[batch_size]) * 0.001,
                         T.zeros([batch_size, D])),
                        ))

    if smooth:
        def kalman_smooth(previous, potential):
            t, prev = previous

            mat_inv = T.einsum('ab,ibc->iac', J_21[t], T.matrix_inverse(potential[0] + J_11[t][None]))
            J_t1_t = J_22[t][None] - T.einsum('iab,bc->iac', mat_inv, J_12[t])
            h_t1_t = h2[t] - T.einsum('iab,ib->ib', mat_inv, potential[1] + h1[t])

            mat_inv = T.einsum('ab,ibc->iac', J_12[t], T.matrix_inverse(prev[0] - J_t1_t +
                            J_22[t][None]))
            J_tT = potential[0] + J_11[t][None] - T.einsum('ab,ibc,dc->iad',
                                                        J_12[t], mat_inv, J_12[t])
            h_tT = potential[1] + h1[t] - T.einsum('ab,ibc,ic->ia',
                                                        J_12[t], mat_inv,
                                                prev[1] - h_t1_t + h2[t])
            return t + 1, (J_tT, h_tT)
        _, smoothed = T.core.scan(kalman_smooth, (filtered[0][:-1],
                                             filtered[1][:-1]),
                             (0, (filtered[0][-1], filtered[1][-1])),
                             reverse=True)
        smoothed = (
            T.concatenate([smoothed[0], filtered[0][-1:]], 0),
            T.concatenate([smoothed[1], filtered[1][-1:]], 0),
        )
    else:
        smoothed = filtered


    return stats.Gaussian([
        T.matrix_inverse(smoothed[0]),
        T.matrix_solve(smoothed[0], smoothed[1][..., None])[..., 0]
    ])
