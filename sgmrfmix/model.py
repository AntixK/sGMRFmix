import numpy as np
import _sgmrfmix as sgm

class sGMRFmix:

    def __init__(self,
                 K:int,
                 rho:float,
                 m0:np.ndarray=None,
                 pi_threshold:float=0.01,
                 lambda0:float=1.0,
                 max_iter:int=500,
                 tol:float=1e-1,
                 verbose:bool= False,
                 random_seed:int=69):
        self.K = K
        self.rho = rho
        self.pi_threshold = pi_threshold
        self.lambda0 = lambda0
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.verbose = verbose
        np.random.seed(random_seed)
        self.m0 = m0

        self.model_param = None

    def __repr__(self):
        pass

    def fit(self, train_data:np.ndarray):
        N, M = train_data.shape
        self.M = M

        if self.m0 is None:
            self.m0 = np.zeros(M)
        results = sgm.sgmrfmix_fit(train_data,
                         self.K,
                         self.rho,
                         self.m0,
                         False,
                         self.pi_threshold,
                         self.lambda0,
                         self.max_iter,
                         self.tol,
                         self.verbose,
                         self.random_seed)

        self.model_param = results

        if self.verbose:
            print(self.model_param)

        return results

    def compute_anomaly(self, test_data:np.ndarray):
        pass
        # N, M = test_data
        #
        # assert M == self.M, f"dim {M} of the test data must the same as that of the train data (dim = {self.M})"
        #
        # results = sgm.compute_anomaly_score()

m = sGMRFmix(5, 0.8)
train = np.genfromtxt('../Examples/train.csv', delimiter=',', skip_header=True)[:, 1:]
test = np.genfromtxt('../Examples/test.csv', delimiter=',', skip_header=True)[:, 1:]
print(train.shape, test.shape)
results = m.fit(train)
print(results)