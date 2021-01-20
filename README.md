# sGMRFmix

Python library for sGMRFmix model for anomaly detection in time-series data.
sGMRFmix is short for sparse mixture of Gaussian Markov Random Fields.
This is essentially a C++ (and python) port of the R package [`sGMRFmix`](https://cran.r-project.org/web/packages/sGMRFmix/index.html) to make it run faster for larger datasets.

## Model Overview
sGMRFmix is a mixture of GMRFs that predict the likelihood of a random variable using the variables in its markov blanket. Lower the log likelihood, higher the anomaly score. The markov blanket is estimated using a Gaussian graphical model with constraint that enforces sparsity in the inverse covariance matrices of the mixture of GMRF model. This can be done in a stright-forward manner using Graphical LASSO model. You can check out the [paper](https://ide-research.net/papers/2016_ICDM_Ide.pdf) for further details and the math.
  
![sGMRFmix Model](https://github.com/AntixK/sGMRFmix/blob/main/assets/model_overview.png)

## Installation

### Binaries

### Build from source (Linux)

<details>
<summary><h3>Build from source (Linux)</h3></summary>
<p>Install the follow dependencies on Ubuntu/Debian using apt.</p>
<pre>
  <code>
    sudo apt update
    sudo apt install openssh-server libarmadillo-dev libboost-all-dev build-essential
  </code>
</pre>
<p>Clone the repository (including the pybind11 submodule) into a suitable directory</p>
<pre>
  <code>
    git clone --recursive git@github.com:AntixK/sGMRFmix.git
    cd sGMRFmix
  </code>
</pre>
<p>Build the C++ files</p>
<pre>
  <code>
    cd cmake-build-debug
    cmake ..
    make
  </code>
</pre>
<p>Install requirements and build the library.<br>
   Optionally create a python virtual environment to install the library.</p>
 <pre>
   <code>
     cd ..
     pip install -r requirements.txt
     python setup.py install
   </code>
 </pre>
</details>

Install the follow dependencies on Ubuntu/Debian using apt.
```
sudo apt update
sudo apt install openssh-server libarmadillo-dev libboost-all-dev build-essential
```
Clone the repository (including the pybind11 submodule) into a suitable directory
```
git clone --recursive git@github.com:AntixK/sGMRFmix.git
cd sGMRFmix
```
Build the C++ files
```
cd cmake-build-debug
cmake ..
make
```
Install requirements and build the library.
Optionally create a python virtual environment to install the library.
```
cd ..
pip install -r requirements.txt
python setup.py install
```

## Usage
```python
import numpy as np
from sgmrfmix import sGMRFmix

m = sGMRFmix(K = 5, rho=0.8)
train = np.genfromtxt('train.csv', delimiter=',', skip_header=True)[:, 1:]
test = np.genfromtxt('test.csv', delimiter=',', skip_header=True)[:, 1:]

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
- https://github.com/JClavel/glassoFast
