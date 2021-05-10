import numpy as np
import cvxpy as cp
import time

from .utils import neural_decomposition


class Convex_FC_Scalar_Solver():
    def __init__(self):
        self.a, self.b, self.c = 0.09, 0.5, 0.47

    def solve(self, X, y, beta, max_iters):
        D = X.shape[1]
        X_V = self.compute_X_V(X)
        Z, Z_prime, loss = self.solve_cvx(X_V, y, D, beta, max_iters)
        return Z.value, Z_prime.value, loss

    def compute_X_V(self, X):
        '''
        @param X: the (N,D) data matrix
        @returns: X_V (X vectorized) a (N, D**2+D+1) matrix
        that will be used to compute y = X_V @ Z_V 
        (which corresponds to y_hat in the objective).
        '''
        a, b, c = self.a, self.b, self.c
        N, D = X.shape
        X_V = np.zeros((N, D**2+D+1))
        for i in range(N):
            x_i = X[i:i+1, :].T
            X_V[i, 0:D**2] = a*(x_i @ x_i.T).reshape(D**2)
            X_V[i, D**2:D**2+D] = b*x_i.reshape(D)
            X_V[i, D**2+D] = c*1

        return X_V

    def solve_cvx(self, X_V, y, D, beta, max_iters):
        Z1 = cp.Variable((D, D), symmetric=True)
        Z2 = cp.Variable((D, 1))
        Z4 = cp.Variable((1, 1))

        Z1_prime = cp.Variable((D, D), symmetric=True)
        Z2_prime = cp.Variable((D, 1))
        Z4_prime = cp.Variable((1, 1))

        Z = cp.vstack([
            cp.hstack((Z1, Z2)),
            cp.hstack((Z2.T, Z4))
        ])

        Z_prime = cp.vstack([
            cp.hstack((Z1_prime, Z2_prime)),
            cp.hstack((Z2_prime.T, Z4_prime))
        ])

        Z_V = cp.vstack([
            cp.reshape((Z1-Z1_prime), (D**2, 1)),
            (Z2-Z2_prime),
            (Z4-Z4_prime)
        ])

        yhat = X_V@Z_V

        loss = 0.5*cp.sum_squares(yhat - y) + beta*(Z4 + Z4_prime)
        # if loss_name == "squared_loss":
        #     objective = 0.5*cp.sum_squares(yhat - y) + beta*(Z4 + Z4_prime)
        # elif loss_name == "l1_loss":
        #     objective = cp.sum(cp.abs(yhat - y)) + beta*(Z4 + Z4_prime)
        # elif loss_name == "huber":
        #     objective = cp.sum(cp.huber(yhat-y)) + beta*(Z4 + Z4_prime)
        objective = cp.Minimize(loss)

        constraints = [
            Z >> 0,
            Z_prime >> 0,
            cp.trace(Z1) == Z4,
            cp.trace(Z1_prime) == Z4_prime,
        ]

        problem = cp.Problem(objective, constraints)
        start_time = time.time()
        print('########')
        print("Started Convex FC Scalar Solver...")
        problem.solve(warm_start=False, max_iters=max_iters)
        end_time = time.time()
        print(f'Finished, time elapsed: {end_time - start_time}')

        # Print result.
        print(f'Status: {problem.status}')
        print(f'Optimal value: {objective.value}')
        print('########')

        return Z, Z_prime, objective.value


def compute_scalar_weights_from_Z(Z, Z_prime, tolerance):
    # each p_j column vector of Z_decomposed and Z_prime_decomposed
    # can be represented by p_j = [c_j^T  d_j] = [u_j^T ||c_j||   d_j]
    # where c_j \in R^D, d_j \in R, and u_j = c_j / ||c_j||.

    # Then W1 = {u_1, ..., u_r, u'_1,...,u'_r'} of shape (r + r', D)
    # and  W2 = {d_1^2, ..., d_r^2, d'_1^2, ..., d'_r'^2} of shape (r + r', 1)

    Z_decomposed = neural_decomposition(Z, tolerance)
    Z_prime_decomposed = neural_decomposition(Z_prime, tolerance)

    c_js = Z_decomposed[:-1, :]  # first D rows
    c_js_prime = Z_prime_decomposed[:-1, :]  # first D rows
    c_js_all = np.hstack([c_js, c_js_prime])  # shape (D, r + r')

    d_js = Z_decomposed[-1:, :]  # last row
    d_js_prime = Z_prime_decomposed[-1:, :]  # last row
    d_js_all = np.hstack([d_js, d_js_prime])  # shape (1, r + r')

    u_js_all = c_js_all / np.sqrt(np.sum(c_js_all**2, axis=0))  # normalize

    W1 = u_js_all * np.sign(d_js_all)           # shape (D, r + r')
    W2 = np.hstack([d_js**2, -d_js_prime**2]).T  # shape (r + r', 1)

    return W1, W2
