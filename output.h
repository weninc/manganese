#include <map>
#include <complex>
#include <string>
#include <vector>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <fftw3.h>

struct Dataset
{
    hid_t id, type, filespace, memspace;
    hsize_t m, n, stride;
};

class File
{
public:
    File(const char* name);
    void addDataset(std::string name, hid_t type, hsize_t m, hsize_t n, hsize_t stride=1);
    void writeRow(std::string name, hsize_t row, void* data);
    void addAttribute(std::string name, double val);
    void addIntAttribute(std::string name, int val);
    ~File();
    hid_t file;
private:
    hid_t para_group;
    std::map<std::string, Dataset> dsets;
}; 

class Spectrum
{
public:
    Spectrum(double fmin, double fmax, int nt, double dt);
    void saveParameters(File& file);
    void compute(std::complex<double>* input, int nt);
    int nelements() {return m_nfreq;}
    double* data() {return m_spectrum.data();}
    ~Spectrum();
private:
    std::complex<double>* m_fft;
    std::vector<double> m_spectrum;
    fftw_plan m_plan;
    int m_start, m_end, m_nfreq;
    double m_fmin, m_fmax;
};
