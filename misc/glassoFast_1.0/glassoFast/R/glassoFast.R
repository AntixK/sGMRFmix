# GLASSO algorithm of Friedman et al. 2008 with FORTRAN implementation of Sustik and Calderhead 2012.
# Ported to R by J. Clavel <julien.clavel@hotmail.fr> / <clavel@biologie.ens.fr> - 2017.

glassoFast <-
function(S, rho, thr=1.0e-4, maxIt=1e4, start=c("cold","warm"), w.init=NULL, wi.init=NULL, trace=FALSE){
  
  n=nrow(S)           # dimension of S
  if(is.matrix(rho)){
      if(length(rho)!=n*n) stop("The input matrix for \"rho\" must be of size ",n," by ",n)
      L = rho         # matrix of regularization parameters
  }else{
      L = matrix(rho,n,n) # matrix of regularization parameters
  }
  
  # cold or warm start
  start.type=match.arg(start)
  if(start.type=="cold"){
    is=0
    W=X=matrix(0,nrow=n,ncol=n)
  }
  if(start.type=="warm"){
    is=1
    if(is.null(w.init) | is.null(wi.init)){
      stop("Warm start specified: w.init and wi.init must be non-null")
    }
    W=w.init
    X=wi.init
  }
  
  Wd = WXj = numeric(n)
  
  msg=1*trace
  info = 0
  mode(n)="integer"
  mode(S)="double"
  mode(L)="double"
  mode(thr)="double"
  mode(maxIt)="integer"
  mode(msg)="integer"
  mode(is)="integer"
  mode(X)="double"
  mode(W)="double"
  mode(info)="integer"
  
  
  LASSO<-.Fortran("glassofast",
                 n,
                 S,
                 L,
                 thr,
                 maxIt,
                 msg,
                 is,
                 X,
                 W,
                 Wd,
                 WXj,
                 info)
  
  results <- list(w=LASSO[[9]], wi=LASSO[[8]], errflag=LASSO[[12]], niter=LASSO[[5]])
  return(results)
}
