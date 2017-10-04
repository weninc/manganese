#include <complex>
#include <cstring>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include "propagator.h"
using namespace std;

Propagator::Propagator(int ns)
{
    nstates = ns;
    eigval = new double[ns];
    M.resize(ns, ns);
    U.resize(ns, ns);
}

void Propagator::propagate(double dt, Matrix& rho, Matrix& H)
{
    const dcomplex alpha(1.0, 0.0);
    const dcomplex beta(0.0, 0.0);
    
    // eigenvalue decomposition
    int ret  = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'L', 
                              nstates, H.data, nstates, eigval);
    if (ret != 0) {
        cout<<"Error in getting eigenvalues"<<endl;
        exit(-1);
    }
    
    // copy eigenvectors into new matrix
    cblas_zcopy(nstates*nstates, H.data, 1, M.data, 1);
    
    // mulitply each colum with exponential of the eigenvalue
    for (int i=0; i<nstates; i++) {
        dcomplex exponential = exp(-dcomplex(0.0, 1.0)*eigval[i]*dt);
        cblas_zscal(nstates, &exponential, &M(0, i), 1);
    }
    // U = M * eigvec
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nstates, nstates, nstates,
                &alpha, M.data, nstates, H.data, nstates, &beta, U.data, nstates);
    
    // do unitary time evolution
    // rho^i+1 = U * rho * U^*
    cblas_zhemm(CblasColMajor, CblasRight, CblasLower, nstates, nstates, 
                &alpha, rho.data, nstates, U.data, nstates, &beta, M.data, nstates);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nstates, nstates, nstates,
                &alpha, M.data, nstates, U.data, nstates, &beta, rho.data, nstates);
} 
