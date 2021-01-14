# glassoFast
glassoFast: a Fast Graphical LASSO

This package propose a fast implementation of the graphical LASSO of Friedman et al. 2008 based on the algorithm (and FORTRAN subroutine) of Sustik and Calderhead (2012).
This algorithm also avoid non-termination issues observed for the "glasso" function of the R package glasso.


**glassoFast 1.0.0**

1. This is the version 1.0.0:
 
## **Package Installation**


You can install it directly from gitHub through devtools:

```
library(devtools)

install_github("JClavel/glassoFast")

```


(Note that you may also need to install Rtools to compile the C and FORTRAN codes included in the package. For [Windows] (https://cran.r-project.org/bin/windows/Rtools/) and for [Mac] (http://r.research.att.com) (and [Tools] (https://r.research.att.com/tools/). See also "gcc" and "gfortran" websites)

## **Report an issue**
Any bugs encountered when using the package can be reported [here](https://github.com/JClavel/glassoFast/issues)

## **References**
**Friedman J., Hastie T., Tibshirani R. 2008.** Sparse inverse covariance estimation with the graphical lasso. Biostatistics. 9:432-441.

**Sustik M.A., Calderhead B. 2012.** GLASSOFAST: An efficient GLASSO implementation. UTCS Technical Report TR-12-29:1-3. [Source code](http://www.cs.utexas.edu/users/sustik/glassofast/)

