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
                 verbose:bool= True,
                 random_seed:int=314):
        self.K = K
        self.rho = rho
        self.pi_threshold = pi_threshold
        self.lambda0 = lambda0
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.verbose = verbose
        np.random.seed(random_seed)
        # np.printoptions(precision=2, supress=True)

        self.m0 = m0
        self.do_kmeans = False

        self.model_param = {}
        self.model_param['precision_matrices'] = None
        self.model_param['mean_vectors'] = None
        self.model_param['gating_matrix'] = None

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
                         self.do_kmeans,
                         self.pi_threshold,
                         self.lambda0,
                         self.max_iter,
                         self.tol,
                         self.verbose,
                         self.random_seed)

        self.model_param['precision_matrices'] = results[0]
        self.model_param['mean_vectors'] = results[1]
        self.model_param['gating_matrix'] = results[2]

    def compute_anomaly(self, test_data:np.ndarray):
        N, M = test_data.shape

        assert M == self.M, f"dim {M} of the test data must the same as that of the train data (dim = {self.M})"

        results = sgm.compute_anomaly_(test_data,
                                      self.model_param['precision_matrices'],
                                      self.model_param['mean_vectors'],
                                      self.model_param['gating_matrix'],
                                      self.verbose)[0]
        return results

    def show_model_params(self):
        print("sGMRFmix Parameters ==============================")
        for key, val in self.model_param.items():
            print(key + ':')
            print(val)

        print("==================================================")
