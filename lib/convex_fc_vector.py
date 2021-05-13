import numpy as np
import cvxpy as cp
import time

from .utils import neural_decomposition


class Convex_FC_Vector_Solver():
    def __init__(self):
        self.a, self.b, self.c = 0.09, 0.5, 0.47

    def get_Y_expanded(self, y, C):
        ''' 
        @param y -> (N,)
        @return Y -> (N, C) such that
        Y[i, j] = 1 if y[i] == j else 0
        '''
        N = y.shape[0]
        Y = np.zeros((N, C))
        Y[range(N), y] = 1
        return Y

    def solve(self, X, y, num_classes, beta, max_iters, verbose=False):
        '''
        X -> (N, D)
        Y -> (N, C)
        '''
        N, D = X.shape
        C = num_classes
        a, b, c = self.a, self.b, self.c
        Z1, Z2, Z4, Z1_prime, Z2_prime, Z4_prime = [], [], [], [], [], []
        Z, Z_prime, Z_V = [], [], []

        Y = self.get_Y_expanded(y, C)
        print(f'Y.shape = {Y.shape}')  # (N, C)

        for k in range(C):
            Z1.append(cp.Variable((D, D), symmetric=True))
            Z2.append(cp.Variable((D, 1)))
            Z4.append(cp.Variable((1, 1)))

            Z1_prime.append(cp.Variable((D, D), symmetric=True))
            Z2_prime.append(cp.Variable((D, 1)))
            Z4_prime.append(cp.Variable((1, 1)))

            Z.append(cp.vstack([
                cp.hstack((Z1[k], Z2[k])),
                cp.hstack((Z2[k].T, Z4[k]))]))

            Z_prime.append(cp.vstack([
                cp.hstack((Z1_prime[k], Z2_prime[k])),
                cp.hstack((Z2_prime[k].T, Z4_prime[k]))]))

        Y_hat = cp.Variable((N, C))
        constraints = []

        for j in range(N):
            x_j = X[j:j+1, :].T  # shape (D,1)
            for k in range(C):
                quad_term = a * (x_j.T @ (Z1[k] - Z1_prime[k]) @ x_j)
                linear_term = b * (x_j.T @ (Z2[k] - Z2_prime[k]))
                const_term = c * (Z4[k] - Z4_prime[k])

                constraints += [Y_hat[j, k] ==
                                quad_term + linear_term + const_term]

        for k in range(C):
            constraints += [
                Z[k] >> 0,
                Z_prime[k] >> 0,
                cp.trace(Z1[k]) == Z4[k],
                cp.trace(Z1_prime[k]) == Z4_prime[k],
            ]

        loss = 0.5 * cp.sum_squares(Y - Y_hat) + beta * cp.sum(Z4 + Z4_prime)
        objective = cp.Minimize(loss)

        problem = cp.Problem(objective, constraints)

        start = time.time()
        print('########')
        print("Started Convex FC Vector Solver...")
        problem.solve(warm_start=False, max_iters=max_iters, verbose=verbose)
        end = time.time()
        print(f'Finished, time elapsed: {end - start}')

        # Print result.
        print(f'Status: {problem.status}')
        print(f'Optimal value: {objective.value}')
        print('########')

        Z_values = [Z[k].value for k in range(C)]
        Z_prime_values = [Z_prime[k].value for k in range(C)]
        return Z_values, Z_prime_values, objective.value

    def predict(self, X, Z, Z_prime):
        a, b, c = self.a, self.b, self.c
        N, D = X.shape
        C = len(Z)  # = len(Z_prime)
        Y_hat = np.zeros((N, C))
        for j in range(N):
            x_j = X[j:j+1, :].T  # shape (D,1)
            for k in range(C):
                quad_term = a * \
                    (x_j.T @ (Z[k][:D, :D] - Z_prime[k][:D, :D]) @ x_j)
                linear_term = b * (x_j.T @ (Z[k][:D, D:] - Z_prime[k][:D, D:]))
                const_term = c * (Z[k][D, D] - Z_prime[k][D, D])
                Y_hat[j, k] = quad_term + linear_term + const_term
        return Y_hat

    def calculate_loss(self, Y_hat, Y, Z, Z_prime, beta, D):
        # TODO: Fix Y
        _, C = Y.shape
        return 0.5 * np.sum((Y - Y_hat)**2) + beta * np.sum([[Z[k][D, D] + Z_prime[k][D, D]] for k in range(C)])


class Convex_FC_Vector_Solver_Vectorized():
    def __init__(self):
        self.a, self.b, self.c = 0.09, 0.5, 0.47

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

    def get_Y_expanded(self, y, C):
        '''
        @param y -> (N,)
        @return Y -> (N, C) such that
        Y[i, j] = 1 if y[i] == j else 0
        '''
        N = y.shape[0]
        Y = np.zeros((N, C))
        Y[range(N), y] = 1
        return Y

    def solve(self, X, y, num_classes, beta, max_iters, verbose=False):
        '''
        X -> (N, D)
        Y -> (N, C)
        '''
        N, D = X.shape
        C = num_classes
        Z1, Z2, Z4, Z1_prime, Z2_prime, Z4_prime = [], [], [], [], [], []
        Z, Z_prime, Z_V = [], [], []

        Y = self.get_Y_expanded(y, C)
        print(f'Y.shape = {Y.shape}')  # (N, C)

        X_V = self.compute_X_V(X)
        print(f'X_V.shape = {X_V.shape}')  # (N, D**2+D+1)

        for k in range(C):
            Z1.append(cp.Variable((D, D), symmetric=True))
            Z2.append(cp.Variable((D, 1)))
            Z4.append(cp.Variable((1, 1)))

            Z1_prime.append(cp.Variable((D, D), symmetric=True))
            Z2_prime.append(cp.Variable((D, 1)))
            Z4_prime.append(cp.Variable((1, 1)))

            Z.append(cp.vstack([
                cp.hstack((Z1[k], Z2[k])),
                cp.hstack((Z2[k].T, Z4[k]))]))

            Z_prime.append(cp.vstack([
                cp.hstack((Z1_prime[k], Z2_prime[k])),
                cp.hstack((Z2_prime[k].T, Z4_prime[k]))]))

        Z_V = cp.hstack([cp.vstack([
            cp.reshape((Z1[k]-Z1_prime[k]), (D**2, 1)),
            (Z2[k]-Z2_prime[k]),
            (Z4[k]-Z4_prime[k])
        ]) for k in range(C)])

        print(f'Z_V.shape = {Z_V.shape}')  # (D**2+D+1, C)

        Y_hat = X_V @ Z_V

        loss = 0.5 * cp.sum_squares(Y - Y_hat) + beta * cp.sum(Z4 + Z4_prime)

        objective = cp.Minimize(loss)

        constraints = []
        for k in range(C):
            constraints += [
                Z[k] >> 0,
                Z_prime[k] >> 0,
                cp.trace(Z1[k]) == Z4[k],
                cp.trace(Z1_prime[k]) == Z4_prime[k],
            ]

        problem = cp.Problem(objective, constraints)

        start = time.time()
        print('########')
        print("Started Convex FC Vector Solver...")
        problem.solve(warm_start=False, max_iters=max_iters, verbose=verbose)
        end = time.time()
        print(f'Finished, time elapsed: {end - start}')

        # Print result.
        print(f'Status: {problem.status}')
        print(f'Optimal value: {objective.value}')
        print('########')

        Z_values = [Z[k].value for k in range(C)]
        Z_prime_values = [Z_prime[k].value for k in range(C)]
        return Z_values, Z_prime_values, objective.value

    def predict(self, X, Z, Z_prime):
        _, D = X.shape
        C = len(Z)  # = len(Z_prime)
        X_V = self.compute_X_V(X)
        Z_V = np.hstack([np.vstack([
            np.reshape((Z[k][:D, :D]-Z_prime[k][:D, :D]), (D**2, 1)),
            (Z[k][:D, D:]-Z_prime[k][:D, D:]),
            (Z[k][D, D]-Z_prime[k][D, D])
        ]) for k in range(C)])
        Y_hat = X_V @ Z_V
        return Y_hat

    def calculate_loss(self, Y_hat, Y, Z, Z_prime, beta, D):
        # TODO: Fix Y
        _, C = Y.shape
        return 0.5 * np.sum((Y - Y_hat)**2) + beta * np.sum([[Z[k][D, D] + Z_prime[k][D, D]] for k in range(C)])

