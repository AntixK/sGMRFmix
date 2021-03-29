import unittest
import numpy as np
from sgmrfmix import sGMRFmix

class TestsGMRFmix(unittest.TestCase):
    def setUp(self) -> None:
        self.m = sGMRFmix(K=5, rho=0.8)
        self.train = np.genfromtxt('../Examples/Data/train.csv', delimiter=',', skip_header=True)[:, 1:]
        self.test = np.genfromtxt('../Examples/Data/test.csv', delimiter=',', skip_header=True)[:, 1:]

    def test_model(self):
        # print(train.shape, test.shape)
        self.m.fit(self.train)
        self.m.show_model_params()
        results = self.m.compute_anomaly(self.test)

        # print("anomaly score:")
        # print(results)
        # # print([r.shape for r in results])
        # plt.plot(results[:,0])
        # plt.show()