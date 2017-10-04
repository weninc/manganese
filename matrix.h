#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

// Column major complex matrix class
class Matrix
{
public:
    Matrix()
    {
        m = 0;
        n = 0;
        data = 0;
    }
    Matrix(int _m, int _n) : m(_m), n(_n)
    {
        int ret = posix_memalign((void**)&data, 32, sizeof(lapack_complex_double)*m*n);
        if (ret != 0) {
            std::cout<<"Error in allocating matrix!"<<std::endl;
        }
    }
    ~Matrix()
    {
        free(data);
    }
    void resize(int _m, int _n)
    {
        m = _m;
        n = _n;
        int ret = posix_memalign((void**)&data, 32, sizeof(lapack_complex_double)*m*n);
        if (ret != 0) {
            std::cout<<"Error in allocating matrix!"<<std::endl;
        }
        
    }
    inline lapack_complex_double& operator () (int i, int j)
    {
        assert((i<m) && (j<n));
        return data[i + j*m];
    }
    void setZeros()
    {
        for (int i=0; i<m*n; i++) {
            data[i] = 0.0;
        }
    }
    lapack_complex_double* data;
private:
    int m, n;  
}; 

#endif // MATRIX_H