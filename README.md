# sGMRFmix

Python library for sGMRFmix model for anomaly detection in time-series data.
sGMRFmix is short for sparse mixture of Gaussian Markov Random Fields.

## Model Overview

## Installation

### Pre-Requisities

Install the follow dependencies on Ubuntu/Debian using apt.
```
sudo apt install libarmadillo-dev
sudo apt install libboost-all-dev
```

### Build from source

## pip installation

## Usage
```python
import numpy as np
from sgmrfmix import sGMRFmix

model = sGMRFmix()
model.fit(X_train)
anomaly_score = model.compute_anomaly(X_test)
```

Check out further examples in the `Examples/` folder.


## References
