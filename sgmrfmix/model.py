import numpy as np
import _sgmrfmix as sgm
import pickle
import pprint

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
                 random_seed:int=314,
                 **kwargs):
        self.K = K
        self.rho = rho
        self.m0 = m0
        self.pi_threshold = pi_threshold
        self.lambda0 = lambda0
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_seed = random_seed
        np.random.seed(random_seed)
        # np.printoptions(precision=2, supress=True)

        self.do_kmeans = False

        self.model_param = {}
        self.model_param['precision_matrices'] = None
        self.model_param['mean_vectors'] = None
        self.model_param['gating_matrix'] = None


    def __repr__(self):
        return self.__class__.__name__ + \
               f"({','.join([f'{k} = {v}' for i, (k,v) in enumerate(self.__dict__.items()) if i < 9])})"

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

    def get_params(self):
        return self.model_param

    def save(self, filename:str):
        # print("Saving sGMRFmix Model ============================")

        assert filename.split('.')[1] == "pkl", "File extension must be a pkl"

        with open(filename, 'wb') as file:
            pickle.dump(self.model_param, file)

    def load(self, filename:str):
        with open(filename, "rb") as fp:
            params = pickle.load(fp)

        # Check for the relevant keys
        assert 'precision_matrices' in params, f"Given {filename} does not contain 'precision_matrices' key"
        assert 'mean_vectors' in params, f"Given {filename} does not contain 'mean_vectors' key"
        assert 'gating_matrix' in params, f"Given {filename} does not contain 'gating_matrix' key"

        self.model_param = params

