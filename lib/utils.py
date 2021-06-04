import numpy as np

def relative_error(x, y):
  """ returns relative error """
  assert x.shape == y.shape
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def flatten(x):
    return x.reshape((x.shape[0], -1))


def neural_decomposition(Z_decomp, tolerance):
    '''
    We just take this verbatim from Burak's code.
    See sec. 4.1 in paper.


    Decomposes Z as a sum of r=rank(Z) rank 1 matrices.
    Z = \sum_{j=1}^r y_j^T @ y_j, for j \in [1..r]
    such that y_j^T @ G @ y_j = 0.

    @returns A matrix of shape (D+1, r) such that every column
    is a decomposed vector
    '''
    G = np.identity(Z_decomp.shape[0])
    G[-1, -1] = -1

    # step 0
    evals, evecs = np.linalg.eigh(Z_decomp)
    # some eigvals are negative due to numerical issues, tolerance masking deals with that
    ind_pos_evals = (evals > tolerance)
    p_i_all = evecs[:, ind_pos_evals] * np.sqrt(evals[ind_pos_evals])

    outputs_y = np.zeros(p_i_all.shape)

    for i in range(outputs_y.shape[1]-1):
        # step 1
        p_1 = p_i_all[:, 0:1]
        p_1Gp_1 = np.matmul(p_1.T, np.matmul(G, p_1))

        if p_1Gp_1 == 0:
            y = p_1.copy()

            # update
            p_i_all = np.delete(p_i_all, 0, 1)  # delete the first column
        else:
            for j in range(1, p_i_all.shape[1]):
                p_j = p_i_all[:, j:j+1]
                p_jGp_j = np.matmul(p_j.T, np.matmul(G, p_j))
                if p_1Gp_1 * p_jGp_j < 0:
                    break

            # step 2
            p_1Gp_j = np.matmul(p_1.T, np.matmul(G, p_j))
            discriminant = 4*p_1Gp_j**2 - 4*p_1Gp_1*p_jGp_j
            alpha = (-2*p_1Gp_j + np.sqrt(discriminant)) / (2*p_jGp_j)
            y = (p_1 + alpha*p_j) / np.sqrt(1+alpha**2)

            # update
            p_i_all = np.delete(p_i_all, j, 1)  # delete the jth column
            p_i_all = np.delete(p_i_all, 0, 1)  # delete the first column

            u = (p_j - alpha*p_1) / np.sqrt(1+alpha**2)
            # insert u to the list of p_i's
            p_i_all = np.concatenate((p_i_all, u), axis=1)

        # save y
        outputs_y[:, i:i+1] = y.copy()

    # save the remaining column
    outputs_y[:, -1:] = p_i_all.copy()

    return outputs_y
