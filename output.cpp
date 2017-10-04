#include <iostream>
#include <cmath>
#include <cstdlib>
#include "output.h"
using namespace std;

File::File(const char* name)
{
    file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    para_group = H5Gcreate(file, "/para", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

void File::addDataset(string name, hid_t type, hsize_t m, hsize_t n, hsize_t stride)
{
    Dataset dset;
    dset.type = type;
    dset.m = m;
    dset.n = n;
    dset.stride = stride;
    hsize_t dims_dset[] = {m, n};
    dset.filespace = H5Screate_simple(2, dims_dset, NULL);
    
    // memspace is the dataspace
    hsize_t dims_memspace[] = {dset.n*dset.stride};
    dset.memspace = H5Screate_simple(1, dims_memspace, NULL);
    dset.id = H5Dcreate(file, name.c_str(), type, dset.filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dsets.insert(pair<string, Dataset>(name, dset));
}

void File::writeRow(string name, hsize_t row, void* data)
{
    Dataset dset = dsets.at(name);
    if (row >= dset.m) {
        cout<<"Row index "<<row<<" bigger then number of rows  "<<dset.m<<endl;
        return;
    }
    hsize_t dims[] = {1, dset.n};
    hsize_t offset[] = {row, 0};
    // select row in the 2d filespace
    H5Sselect_hyperslab(dset.filespace, H5S_SELECT_SET, offset, NULL, dims, NULL);
    
    hsize_t start[] = {0};
    hsize_t stride[] = {dset.stride};
    hsize_t count[] = {dset.n};
    H5Sselect_hyperslab(dset.memspace, H5S_SELECT_SET, start, stride, count, NULL);
    
    H5Dwrite(dset.id, dset.type, dset.memspace, dset.filespace, H5P_DEFAULT, data);
}

void File::addAttribute(string name, double val)
{
    H5LTset_attribute_double(file, "/para", name.c_str(), &val, 1);
}

void File::addIntAttribute(string name, int val)
{
    H5LTset_attribute_int(file, "/para", name.c_str(), &val, 1);
}


File::~File()
{
    map<string, Dataset>::const_iterator iter;
    for (iter=dsets.begin(); iter != dsets.end(); iter++) {
        H5Dclose(iter->second.id);
        H5Sclose(iter->second.memspace);
        H5Sclose(iter->second.filespace);
    }
    H5Fclose(file);
}

Spectrum::Spectrum(double fmin, double fmax, int nt, double dt)
{
    double dw = 2.0*M_PI / ((nt-1)*dt) * 27.211;
    int middle = nt / 2;
    int imin = fmin / dw;
    int imax = fmax / dw;
    m_start = middle + imin;
    m_end = middle + imax;
    m_nfreq = m_end - m_start;
    m_spectrum.resize(m_nfreq);
    
    m_fmin = imin*dw;
    m_fmax = (imax-1)*dw;
   
    int ret = posix_memalign((void**)&m_fft, 16, nt*sizeof(complex<double>));
    if (ret != 0) {
        cout<<"Error in posix_memalign for fft!"<<endl;
        cout<<"Abort"<<endl;
        exit(-1);
    }
    m_plan = fftw_plan_dft_1d(nt, reinterpret_cast<fftw_complex*>(m_fft), 
                            reinterpret_cast<fftw_complex*>(m_fft),
                            FFTW_FORWARD, FFTW_ESTIMATE);    
}

void Spectrum::saveParameters(File& file)
{
    file.addIntAttribute("nfreq", m_nfreq);
    file.addAttribute("fmin", m_fmin);
    file.addAttribute("fmax", m_fmax);
}


void Spectrum::compute(complex<double>* input, int nt)
{
    // we need spectrum for the complex conjugate because of expansion 
    // E(t)*exp(-i omega T)
    for (int i=0; i<nt; i++) {
        m_fft[i] = conj(input[i]);
    }
    
    // shift zero frequency to center by multiplying input with -1^i
    for (int i=1; i<nt; i+=2) {
        m_fft[i] = -m_fft[i];
    }
    
    fftw_execute(m_plan);
    
    int index = 0;
    for (int i=m_start; i<m_end; i++) {
        m_spectrum[index] = norm(m_fft[i]);
        index++;
    }
}

Spectrum::~Spectrum()
{
    free(m_fft);
    fftw_destroy_plan(m_plan);
}



// Test output
/*
int main()
{
    File file("test.h5");
    double data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double data2[] = {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    file.addDataset("data", H5T_NATIVE_DOUBLE, 2, 4, 2);
    file.writeRow("data", 0, data1);
    file.writeRow("data", 1, data2);
}
*/