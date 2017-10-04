#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include <vector>
#include <complex>
#include "matrix.h"

typedef std::complex<double> dcomplex;

class Propagator
{
public:
    Propagator(int ns);
    void propagate(double dt, Matrix& rho, Matrix& H);
private:
    int nstates;
    double* eigval;
    Matrix M, U;
}; 

#endif // PROPAGATOR_H
