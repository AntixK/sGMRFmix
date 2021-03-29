import numpy as np
from sgmrfmix import sGMRFmix


def check_model() -> None:
    m = sGMRFmix(K=5, rho=0.8)
    train = np.genfromtxt('../Examples/Data/train.csv', delimiter=',', skip_header=True)[:, 1:]
    test = np.genfromtxt('../Examples/Data/test.csv', delimiter=',', skip_header=True)[:, 1:]
    print(m)

# def test_model(self):
    print(train.shape, test.shape)
    m.fit(train)
    m.show_model_params()
    results = m.compute_anomaly(test)

    print("Anomaly score:")
    print(results)

    m.save('test_model.pkl')
    m.load('test_model.pkl')
    results2 = m.compute_anomaly(test)
    print(np.allclose(results, results2))
    # # print([r.shape for r in results])
    # plt.plot(results[:,0])
    # plt.show()

check_model()