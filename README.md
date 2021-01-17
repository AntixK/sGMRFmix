# sGMRFmix

Python library for sGMRFmix model for anomaly detection in time-series data.
sGMRFmix is short for sparse mixture of Gaussian Markov Random Fields.
This is essentially a C++ (and python) port of the R package [`sGMRFmix`](https://cran.r-project.org/web/packages/sGMRFmix/index.html) to make it run faster for larger datasets.

## Model Overview
![sGMRFmix Model](https://github.com/AntixK/sGMRFmix/blob/main/assets/model_overview.png)

## Installation

### Pre-Requisities

Install the follow dependencies on Ubuntu/Debian using apt.
```
sudo apt install libarmadillo-dev
sudo apt install libboost-all-dev
```

### Build from source

### pip installation

## Usage
```python

import numpy as np
from sgmrfmix import sGMRFmix

m = sGMRFmix(K = 5, rho=0.8)
train = np.genfromtxt('../Examples/train.csv', delimiter=',', skip_header=True)[:, 1:]
test = np.genfromtxt('../Examples/test.csv', delimiter=',', skip_header=True)[:, 1:]

m.fit(train)
m.show_model_params()
results = m.compute_anomaly(test)
```

Check out further examples in the `Examples/` folder.


## Acknowledgements
- T. Ide, A .Khandelwal, J .Kalagnanam, **Sparse Gaussian Markov Random Field Mixtures for Anomaly Detection**, IEEE 16th International Conference on Data Mining (ICDM), 2016, pp 955–960
- https://rdrr.io/cran/sGMRFmix/f/vignettes/sGMRFmix.Rmd
- https://cran.r-project.org/web/packages/sGMRFmix/vignettes/sGMRFmix.html
- https://github.com/cran/sGMRFmix
