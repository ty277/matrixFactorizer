from matrixFactorizer import MatrixFactorizer
import numpy as np
import unittest
import logging
log = logging.getLogger()
handler = logging.StreamHandler()
FORMAT = "[%(asctime)s:%(filename)s:%(lineno)s]:%(levelname)s: %(message)s"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATETIME_FORMAT)

handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)


def create_random_input_data(N, M, K, missing_rate):
    '''
    input:
        N, M - dimension of original matrix
        missing_rate - 0.0 to 1.0 proportion of elements missing in matrix
    return:
        trueR - true matrix R
        R - observed R with some missing elements
        missing_indices - indices (1d) of missing items
    '''
    # synthesize data
    trueP = np.random.randint(-10, 11, (N, K)) * 1.
    trueQ = np.random.randint(-5, 6, (K, M)) * 1.
    trueR = trueP.dot(trueQ)

    # artificially create missing elements
    R = np.copy(trueR)
    num_missing = int(missing_rate * N * M)
    missing_indices = np.random.choice(N * M, num_missing, replace=False)
    for m in missing_indices:
        row = m // M
        col = m % M
        R[row, col] = np.nan
    return trueR, R


def transform_matrix_to_data(mtx):
    '''
    transform non missing matrix elements into a list of tuples: [(i, j, m_ij), ...]
    '''
    # turn into a data set with known elements
    input_lst = []
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            if np.isnan(mtx[i, j]) == False:
                input_lst.append([i, j, mtx[i, j]])
    return input_lst



class MatrixFactorizerTestCase(unittest.TestCase):
    def setUp(self):
        super(MatrixFactorizerTestCase, self).setUp()
        self.mf = MatrixFactorizer()

    def test_imputing_missing(self):
        '''
        test missing value imputation
        '''
        # make some synthesized data with known ground truth
        N = 20
        M = 30
        K = 5
        missing_rate = 0.2
        trueR, R = create_random_input_data(N, M, K, missing_rate)
        input_data = transform_matrix_to_data(R)
        # use the class to approximate matrix R
        self.mf.train(input_data=input_data, N=N, M=M, K=K)
        R_hat = self.mf.get_R_est()
        # get estimation error of missing terms
        missing_err = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if np.isnan(R[i, j]):
                    missing_err.append(trueR[i, j] - R_hat[i, j])
        err_bar = np.mean(missing_err)
        err_std = np.std(missing_err)
        test_expr = (err_bar + 1.96*err_std > 0.) and (err_bar - 1.96*err_std < 0.)
        self.assertTrue(test_expr, 'Failed inputing missing value test. err_bar: {}, err_std: {}'.format(err_bar, err_std))

    def test_approx_known_matrix(self):
        '''
        test approximation of matrix
        '''
        # make some synthesized data with known ground truth
        N = 20
        M = 30
        K = 5
        missing_rate = 0.
        trueR, _ = create_random_input_data(N, M, K, missing_rate)
        input_data = transform_matrix_to_data(trueR)
        # use the class to approximate matrix R
        self.mf.train(input_data=input_data, N=N, M=M, K=K)
        R_hat = self.mf.get_R_est()

        # get estimation error of missing terms
        err = []
        for i in range(trueR.shape[0]):
            for j in range(trueR.shape[1]):
                err.append(trueR[i, j] - R_hat[i, j])
        test_expr = max(abs(np.percentile(err, 5)), abs(np.percentile(err, 95))) < 0.5
        self.assertTrue(test_expr, 'Failed approximating known matrix test.')




if __name__ == "__main__":
    unittest.main()
