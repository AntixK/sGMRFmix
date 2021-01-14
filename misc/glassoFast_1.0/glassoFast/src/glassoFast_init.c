#include <R_ext/RS.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .Fortran calls */
extern void F77_NAME(glassofast)(int *n, double *S, double *L, double *thr, int *maxIt, int *msg, int *warm, double *X, double *W, double *Wd, double *WXj, int *info);

static const R_FortranMethodDef FortranEntries[] = {
    {"glassofast", (DL_FUNC) &F77_NAME(glassofast), 12},
    {NULL, NULL, 0}
};

void R_init_glassoFast(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, NULL, FortranEntries, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
