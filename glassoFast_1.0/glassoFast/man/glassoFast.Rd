\name{glassoFast}
\alias{glassoFast}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fast graphical LASSO
%%  ~~function to do ... ~~
}
\description{
This function is a faster alternative to the "glasso" function in the glasso package (Friedman et al. 2008). This package is based on the algorithm (FORTRAN subroutine) of Sustik and Calderhead (2012).
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
glassoFast(S, rho, thr = 1e-04, maxIt = 10000, start = c("cold", "warm"), 
w.init = NULL, wi.init = NULL, trace = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{S}{
%%     ~~Describe \code{S} here~~
Covariance matrix (a p by p symmetric matrix)
}
  \item{rho}{
The regularization parameter for lasso. (a non-negative value or a p by p matrix of regularization parameters)
%%     ~~Describe \code{rho} here~~
}
  \item{thr}{
Threshold for convergence. Default is 1.e-4.
%%     ~~Describe \code{thr} here~~
}
  \item{maxIt}{
Maximum number of iterations of outer loop. Default is 10,000.
%%     ~~Describe \code{maxIt} here~~
}
  \item{start}{
Type of start. Cold start is default. Using warm start, can provide starting values for w and wi.
%%     ~~Describe \code{start} here~~
}
  \item{w.init}{
Optional starting values for estimated covariance matrix (p by p). Only needed when start="warm" is specified

%%     ~~Describe \code{w.init} here~~
}
  \item{wi.init}{
Optional starting values for estimated inverse covariance matrix (p by p). Only needed when start="warm" is specified

%%     ~~Describe \code{wi.init} here~~
}
  \item{trace}{
Flag for printing out information as iterations proceed. Default FALSE.
%%     ~~Describe \code{trace} here~~
}
}
\details{
Estimate a sparse inverse covariance matrix using a lasso (L1) penalty, following the Friedman et al. (2008) approach.
The function is a wrapper of the faster and corrected (for non-termination convergence issues) FORTRAN subroutine of Sustik and Calderhead (2012).
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
\item{w }{Estimated covariance matrix}
\item{wi }{Estimated inverse covariance matrix}
\item{errflag }{Memory allocation error flag: 0 means no error; !=0 means memory allocation error}
\item{niter }{Number of iterations of outer loop}
}
\references{
Friedman J., Hastie T., Tibshirani R. 2008. Sparse inverse covariance estimation with the graphical lasso. Biostatistics. 9:432-441.

Sustik M.A., Calderhead B. 2012. GLASSOFAST: An efficient GLASSO implementation. UTCS Technical Report TR-12-29:1-3.


%% ~put references to the literature/web site here ~
}
\author{
Julien Clavel
%%  ~~who you are~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{glasso}
}
\examples{

set.seed(100)

# Make a random covariance matrix
p=5
x<-matrix(rnorm(p*p),ncol=p)
s<- var(x)

# Compute the LASSO estimates
glassoFast(s, rho=.1)

# compare with glasso
\donttest{
 require(glasso)
 glasso(s, rho=.1)

# benchmark glassoFast and glasso
 require(rbenchmark)
 p=100
 x<-matrix(rnorm(p*p),ncol=p)
 s<- var(x)
 benchmark(glassoFast(s, rho=.15), glasso(s, rho=.15), replications = 100)
 # up to an order of magnitude faster
	}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ glasso }
\keyword{ covariance matrix }
\keyword{ regularization }
\keyword{ penalized likelihood }% __ONLY ONE__ keyword per line
