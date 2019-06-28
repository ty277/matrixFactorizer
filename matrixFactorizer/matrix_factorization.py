import numpy as np
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class MatrixFactorizer():
    def __init__(self, N, M, K, lr, lam=0.01, tol=1e-3, maxIter=1000, rnd_seed=54321):
        '''
        Approximating a NxM matrix R by P(NxK) dot Q(KxM) using first order gradient descent.
        Input data comes in as (i, j, R_ij) per row, a row exists only when there is a R_ij value.
        N, M - dimension of the input matrix
        K - dimension of latent variables
        lr - learning rate
        lam - L2 regularization parameter
        maxIter - max number of iterations
        tol - tolerance, early stop if avg squared error change is smaller than this
        rnd_seed - random seed to initialize P, Q
        '''
        # set random seed
        np.random.seed(rnd_seed)
        # initialize MF
        self.K = K
        self.N = N
        self.M = M
        self.lr = lr
        self.lam = lam
        self.tol = tol
        self.maxIter = maxIter
        self.nrows = np.nan  # number of input data rows
        self.loss = np.inf
        self.P = np.nan
        self.Q = np.nan

    def __calc_loss(self, input_lst, P, Q, lam):
        # update loss with new P, Q
        sum_sq_err = 0.
        for i, j, r_ij in input_lst:
            e_ij = r_ij - np.dot(P[i, :], Q[:, j])
            sum_sq_err += e_ij ** 2
            for k in range(P.shape[1]):
                sum_sq_err += 0.5 * lam * (P[i, k] ** 2 + Q[k, j] ** 2)

        return sum_sq_err / self.nrows

    def train(self, input_lst, verbose_log=True):
        '''
        input_lst: iterable of tuples (i, j, R_ij) where i, j is the index in the original matrix,
                    R_ij is the value, usually only contains elements where there is a R_ij value
                    known. e.g. input_data= [(0, 1, 2.5), (0, 2, -10.), (1, 3, -1.), ...]

        '''
        self.nrows = len(input_lst)
        iteration = 0
        self.P = np.random.normal(0., self.K ** -.2, (self.N, self.K))
        self.Q = np.random.normal(0., self.K ** -.2, (self.K, self.M))
        # fit
        while iteration <= self.maxIter:
            # update P, Q
            for i, j, r_ij in input_lst:
                e_ij = r_ij - np.dot(self.P[i, :], self.Q[:, j])  # current error
                for k in range(self.K):
                    self.P[i, k] = self.P[i, k] + self.lr * (2 * e_ij * self.Q[k, j] - self.lam * self.P[i, k])
                    self.Q[k, j] = self.Q[k, j] + self.lr * (2 * e_ij * self.P[i, k] - self.lam * self.Q[k, j])

            # update loss
            loss = self.__calc_loss(input_lst, self.P, self.Q, self.lam)  # updated avg squared error
            delta_loss = self.loss - loss
            self.loss = loss
            iteration += 1

            if verbose_log is True:
                log.info('iteration: {t}, avg. squared loss: {l:.4f}, delta loss:{dl:.4f}'.format(t=iteration,
                                                                                                  l=self.loss,
                                                                                                  dl=delta_loss))
            # check for early stop
            if delta_loss <= self.tol:
                log.info('iteration: {t}, early stop'.format(t=iteration))
                break

    def get_P(self):
        return self.P

    def get_Q(self):
        return self.Q

    def get_R_est(self):
        return np.dot(self.P, self.Q)
