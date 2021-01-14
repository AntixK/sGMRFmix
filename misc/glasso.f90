
program glasso
    integer:: n= 5 
    real::thr = 1e-4
    integer:: maxIt = 10000
    integer:: warm=0
    integer:: msg =0
    
    real, DIMENSION(5, 5) :: X
    real, DIMENSION(5, 5) :: W
    
    real:: Wd(5) = (/ 0, 0, 0, 0, 0 /)
    real:: WXj(5) = (/ 0, 0, 0, 0, 0 /)
    
    real:: S(5,5) = reshape((/ 0.2821, 0.3611,  -0.1907,  -0.1455,  -0.2434, &
       0.3611,   0.6788,  -0.1226,  -0.4604,  -0.1754, &
      -0.1907,  -0.1226,   0.3724,  -0.1257,  -0.0536, &
      -0.1455,  -0.4604,  -0.1257,   0.4503,   0.0712, &
      -0.2434,  -0.1754,  -0.0536,   0.0712,   0.7906 /),(/5, 5/)) 
      
    real:: L(5,5) = reshape((/ 0.1, 0.0, 0.0, 0.0, 0., &
    0.0, 0.1, 0.0, 0.0, 0.0, &
    0.0, 0.0, 0.1, 0.0, 0.0, &
    0.0, 0.0, 0.0, 0.1, 0.0, &
    0.0, 0.0, 0.0, 0.0, 0.1 /), (/5, 5/))
    
    integer iter
    double precision EPS
    parameter (EPS = 1.1e-16)
    info = 0
    shr = sum(abs(S))
    do i = 1,n
       shr = shr - abs(S(i, i))
    enddo
    if (shr .eq. 0.0) then
    !  S is diagonal.
       W = 0.0
       X = 0.0
       do i = 1,n
          W(i,i) = W(i,i) + L(i,i)
       enddo
       X = 0.0
       do i = 1,n
          X(i,i) = 1.0/max(W(i,i),eps)
       enddo
       return
    endif
    shr = thr*shr/(n-1)
    thrLasso = shr/n
    if (thrLasso .lt. 2*EPS) then
       thrLasso = 2*EPS
    end if
    if (warm .eq. 0) then
       W = S
       X = 0.0
    else
       do i = 1,n
         X(1:n,i) = -X(1:n,i)/X(i,i)
         X(i,i) = 0
      end do
    end if
    do i = 1,n
       Wd(i) = S(i,i) + L(i,i)
       W(i,i) = Wd(i)
    end do
    do iter = 1,maxIt
    ! if (msg .ne. 0) write(6,*) "iteration =", iter
    !   print iterations to the console (ok with CRAN)
    ! if (msg .ne. 0)  call intpr('iter:',-1,iter,1)
       dw = 0.0
       do j = 1,n
          WXj(1:n) = 0.0
    !     We exploit sparsity of X when computing column j of W*X*D:
          do i = 1,n
             if (X(i,j) .ne. 0.0) then
                WXj = WXj + W(:,i)*X(i,j)
             endif
          enddo
          do
             dlx = 0.0
             do i = 1,n
                if (i .ne. j) then
                   a = S(i,j) - WXj(i) + Wd(i)*X(i,j)
                   b = abs(a) - L(i,j)
                   if (b .gt. 0.0) then
                      c = sign(b, a)/Wd(i)
                   else
                      c = 0.0
                   endif
                   delta = c - X(i,j)
                   if (delta .ne. 0.0) then
                      X(i,j) = c
                      WXj(1:n) = WXj(1:n) + W(:,i)*delta
                      dlx = max(dlx, abs(delta))
                   endif
                endif
             enddo
             if (dlx .lt. thrLasso) then
                exit
             endif
          enddo
          WXj(j) = Wd(j)
          dw = max(dw, sum(abs(WXj(1:n) - W(:,j))))
          W(:,j) = WXj(1:n)
          W(j,:) = WXj(1:n)
       enddo
    !   write(6,*) "  dw =", dw
       if (dw .le. shr) then
          exit
       endif
    enddo
    do i = 1,n
       tmp = 1/(Wd(i) - sum(X(:,i)*W(:,i)))
       X(1:n,i) = -tmp*X(1:n,i)
       X(i,i) = tmp
    enddo
    do i = 1,n-1
       X(i+1:n,i) = (X(i+1:n,i) + X(i,i+1:n))/2;
       X(i,i+1:n) = X(i+1:n,i) 
    enddo
    maxIt = iter
    
       !display the values
   do i=1,n
      do j = 1, n
         Print *, X(i,j)
      end do
   end do
   
end program