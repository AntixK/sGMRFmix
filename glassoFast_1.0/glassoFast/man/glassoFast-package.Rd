\name{glassoFast-package}
\alias{glassoFast-package}
\docType{package}
\title{
glassoFast: a Fast Graphical LASSO
}
\description{
This package propose a fast implementation of the graphical LASSO of Friedman et al. 2008 based on the algorithm (FORTRAN subroutine) of Sustik and Calderhead (2012).
This algorithm also avoid non-termination issues observed for the "glasso" function of the R package glasso.
}

\details{
\tabular{ll}{
Package: \tab glassoFast\cr
Type: \tab Package\cr
Version: \tab 1.0.0\cr
Date: \tab 2017-06-07\cr
License: \tab GPL (>=3.0)\cr
}

The original FORTRAN Subroutine is available from:

http://www.cs.utexas.edu/users/sustik/glassofast/

}
\author{
Julien Clavel

Maintainer: Julien Clavel <julien.clavel@hotmail.fr>
}

\references{

Friedman J., Hastie T., Tibshirani R. 2008. Sparse inverse covariance estimation with the graphical lasso. Biostatistics. 9:432-441.

Sustik M.A., Calderhead B. 2012. GLASSOFAST: An efficient GLASSO implementation. UTCS Technical Report TR-12-29:1-3.

}

\keyword{ glasso }
\keyword{ Penalized likelihood }
\seealso{
\code{\link{glassoFast}}
}
