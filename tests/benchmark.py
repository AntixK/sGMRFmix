import numpy as np
from time import time
from sgmrfmix import sGMRFmix
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

df_64 = pd.read_csv("../Examples/non_stationary_64.csv",
                    skiprows=[0])

train_array = normalize(df_64.to_numpy())
# print(train_array.shape)

def test_sgmrfmix(N, M, reps = 10):

    x = train_array[:N, :M]
    y = train_array[:N, :M]
    print(N, M)
    times = []
    for _ in range(reps):
        model = sGMRFmix(K=N // 100, rho=0.8, verbose=False)
        model.fit(x)
        start = time()
        results = model.compute_anomaly(y)
        end = time() - start
        times.append(end)

    return np.array(times)

benchmark_results = {}
# test_sgmrfmix(20_000, 410, 2)
N_list = [500, 1_000, 5_000, 10_000, 20_000]
M_list = [5, 25, 100, 250]

for N in tqdm(N_list):
    benchmark_results[f'{N}'] = test_sgmrfmix(N, 5)

# for M in tqdm(M_list):
#     benchmark_results[f'{M}'] = test_sgmrfmix(2000, M)

# with open('py_benchmark.pkl', 'wb') as file:
#     pickle.dump(benchmark_results, file)

with open('py_benchmark_infr.pkl', 'wb') as file:
    pickle.dump(benchmark_results, file)
