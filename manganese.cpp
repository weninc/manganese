#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>
#include <random>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "matrix.h"
#include "output.h"
#include "propagator.h"
using namespace std;

const int NSTATES = 13;

static const double AU_TO_SECONDS = 2.418884326505e-17;
static const double AU_TO_METERS = 5.2917720859e-11;
static const double C = 137.036;

typedef complex<double> dcomplex;

struct Parameters
{
    int nt, nz;
    double dt, dz;
    double omega_svea;
    double density;
    double propagation_length;
    int output_steps;
};

enum PulseType
{
    gaussian,
    sase,
};

enum PulseShape
{
    flattop,
    gaussianShape,
};

struct Pulse
{
    double nph, rfocus, pulse_duration;
    double bandwidth;
    double omega_lcls;
    double rayleigh_length;
    PulseType type;
    PulseShape pshape;
};

struct Atom
{
    double groundstate;
    double sigma1s;
    double auger1s;
    double auger2p;
    double omega[12];
    double dipole[12];
    double gamma_rad[12];
    double sum_gamma_rad;
    double attenuation_solution;
    double reabsorption;
    Matrix rho;
};

void decayStep(Atom& atom, double xfel, dcomplex source[], double dt)
{
    atom.groundstate *= exp(-atom.sigma1s * xfel*0.5*dt);
    // core excited state 1s-1 population
    atom.rho(0, 0) += (atom.sigma1s*xfel*atom.groundstate - (atom.auger1s +atom.sum_gamma_rad)*atom.rho(0, 0))*0.5*dt;

    for (int i=0; i<NSTATES-1; i++) {
        // core-excited state - final state coherences
        atom.rho(i+1, 0) += -0.25*(atom.auger1s + atom.auger2p + atom.gamma_rad[i])*atom.rho(i+1, 0)*dt + source[i];
        // final state decay
        atom.rho(i+1, i+1) -= atom.auger2p*atom.rho(i+1, i+1)*0.5*dt;
    }
}

double gaussian_beam_waist(const double z, const double zr)
{
    double tmp = z / zr;
    return sqrt(1.0 + tmp*tmp);
}

void generate_pulse(const Pulse& pulse, const Parameters& para, mt19937& gen, vector<double>& input)
{
    if (pulse.type  == gaussian) {
        cout<<"Setting up Gaussian pulse"<<endl;
        double sigma = pulse.pulse_duration/(2.0*sqrt(2.0*log(2.0)));
        double pulse_integral = 0.0;
        double fluxfactor = pulse.nph/(pulse.rfocus*pulse.rfocus*M_PI*sigma*sqrt(2.0*M_PI));
        for (int i=0; i<para.nt; i++) {
            double tmp = (i - para.nt/2)*para.dt;
            input[i] = exp(-tmp*tmp / (2.0*sigma*sigma))*fluxfactor;
            pulse_integral += input[i]*para.dt;
        }
        // normalization
        fluxfactor = pulse.nph / (M_PI*pulse.rfocus*pulse.rfocus);
        for (int i=0; i<para.nt; i++) {
            input[i] *= fluxfactor / pulse_integral;
        }
    }
    if (pulse.type == sase) {

        double tcut = 3.0/2.42e-2;
        double sigtaup = (2.0/2.42e-2) / (2.0*sqrt(2.0*log(2.0)));

        double dw = 2.0*M_PI / ((para.nt-1)*para.dt);
        double sigma_bandwidth = pulse.bandwidth/(2.0*sqrt(2.0*log(2.0)));
        dcomplex* zk = (dcomplex*) fftw_malloc(para.nt*sizeof(dcomplex));
        fftw_plan plan = fftw_plan_dft_1d(para.nt,
                                        reinterpret_cast<fftw_complex*>(zk),
                                        reinterpret_cast<fftw_complex*>(zk),
                                        FFTW_BACKWARD, FFTW_ESTIMATE);
        normal_distribution<double> dist;
        for (int i=0; i<para.nt; i++) {
            double w = (i - 0.5*para.nt)*dw;
            double gaussian = exp(-w*w/(2.0*sigma_bandwidth*sigma_bandwidth));
            double real = dist(gen);
            double imag = dist(gen);
            zk[i] = sqrt(gaussian)*dcomplex(real, imag);
        }
        // do inverse fourier transform
        fftw_execute(plan);

        double sigma_duration = pulse.pulse_duration/(2.0*sqrt(2.0*log(2.0)));
        double pulse_integral = 0.0;
        double mask = 0.0;
        for (int i=0; i<para.nt; i++) {
            double t = i*para.dt;
            if (pulse.pshape == gaussianShape) {
                double tmp = t - 3.5*sigma_duration;
                mask = exp(-tmp*tmp / (2.0*sigma_duration*sigma_duration));
            }
            else if (pulse.pshape == flattop) {
                if (t < tcut) {
                    double tmp = t - tcut;
                    mask = exp(-tmp*tmp/(2.0*sigtaup*sigtaup));
                }
                else if (t > pulse.pulse_duration - tcut) {
                    double tmp = t - pulse.pulse_duration + tcut;
                    mask = exp(-tmp*tmp/(2.0*sigtaup*sigtaup));
                }
                else {
                    mask = 1.0;
                }
            }
            input[i] = mask*norm(zk[i]);
            pulse_integral += input[i]*para.dt;
        }
        fftw_destroy_plan(plan);
        fftw_free(zk);

        // normalization
        double fluxfactor = pulse.nph / (M_PI*pulse.rfocus*pulse.rfocus);
        for (int i=0; i<para.nt; i++) {
            input[i] *= fluxfactor / pulse_integral;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout<<"Wrong number of arguments!"<<endl;
        return -1;
    }
    //while (getopt(argc, argv, "n:c:x:o:s:")) != EOF ) {
    //}

    Parameters para;
    para.nt = 10000;
    para.dt = 0.01e-15 / AU_TO_SECONDS;
    // 1M Mn = 6.022 * 10^20 / cm^3 rescale by factor of 10 to make simulation shorter
    para.density = 5.0 * 6.022e20 * 10 * 1.0e6*pow(AU_TO_METERS, 3.0);
    para.output_steps = 100;
    para.propagation_length = 20.0e-6 / AU_TO_METERS;
    para.omega_svea = 5900.0 / 27.211;
    para.dz = C*para.dt;
    para.nz = para.propagation_length / para.dz;
    int step = para.nz / para.output_steps;
    para.output_steps = para.nz / step + 1;
    cout<<"nz:  "<<para.nz<<endl;
    cout<<para.output_steps<<endl;

    Pulse pulse;
    pulse.rfocus = 150.0e-9 / AU_TO_METERS;
    pulse.pulse_duration = 30.0e-15 / AU_TO_SECONDS;
    pulse.bandwidth = 30.0 / 27.211;
    pulse.nph = atof(argv[2]);
    cout<<"nph:  "<<pulse.nph<<endl;
    pulse.rayleigh_length = 10.0e-6 / AU_TO_METERS;
    pulse.type = sase;
    pulse.pshape = flattop;

    Atom atom;

    atom.dipole[0] = 0.008707;
    atom.dipole[1] = 0.007426;
    atom.dipole[2] = 0.005651;
    atom.dipole[3] = 0.004566;
    atom.dipole[4] = 0.003958;
    atom.dipole[5] = 0.003836;
    atom.dipole[6] = 0.003590;
    atom.dipole[7] = 0.002587;
    atom.dipole[8] = 0.002340;
    atom.dipole[9] = 0.002035;
    atom.dipole[10] = 0.001728;
    atom.dipole[11] = 0.001406;

    atom.omega[0] = 5902.3 / 27.211;
    atom.omega[1] = 5901.1 / 27.211;
    atom.omega[2] = 5900.1 / 27.211;
    atom.omega[3] = 5890.3 / 27.211;
    atom.omega[4] = 5899.6 / 27.211;
    atom.omega[5] = 5889.1 / 27.211;
    atom.omega[6] = 5889.1 / 27.211;
    atom.omega[7] = 5890.2 / 27.211;
    atom.omega[8] = 5889.1 / 27.211;
    atom.omega[9] = 5889.0 / 27.211;
    atom.omega[10] = 5890.5 / 27.211;
    atom.omega[11] = 5898.7 / 27.211;

    atom.gamma_rad[0] = 4.0e-04;
    atom.gamma_rad[1] = 2.9e-04;
    atom.gamma_rad[2] = 1.7e-04;
    atom.gamma_rad[3] = 1.1e-04;
    atom.gamma_rad[4] = 8.3e-05;
    atom.gamma_rad[5] = 7.8e-05;
    atom.gamma_rad[6] = 6.8e-05;
    atom.gamma_rad[7] = 3.5e-05;
    atom.gamma_rad[8] = 2.9e-05;
    atom.gamma_rad[9] = 2.2e-05;
    atom.gamma_rad[10] = 1.6e-05;
    atom.gamma_rad[11] = 1.0e-05;

    atom.auger1s = 0.8 / 27.211; // 70% of 1.16 eV K lifetime broadening
    atom.auger2p = 0.32 / 27.211;
    atom.sum_gamma_rad = 0.35 / 27.211; // sum of all gamma_rad

    atom.sigma1s = 3.5e-20*1.0e-4 / (AU_TO_METERS*AU_TO_METERS);

    // Mn 2p photo-ionization cross section
    // atom.sigma2p = 1.8762e-21

    atom.attenuation_solution = AU_TO_METERS / 38.8e-6;

    // reabsorption of Mn K-alpha emission from Cl 1s shell
    atom.reabsorption =  1.3505e-20*1.0e-4 / (AU_TO_METERS*AU_TO_METERS);

    // absorption channels that are always open

    // File output
    // define complex data type for hdf5 output
    hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(dcomplex));
    H5Tinsert(complex_id, "r", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(complex_id, "i", sizeof(double), H5T_NATIVE_DOUBLE);
    File file(argv[1]);
    file.addDataset("field", complex_id, para.output_steps, para.nt);
    file.addDataset("xfel", H5T_NATIVE_DOUBLE, para.output_steps, para.nt);
    file.addDataset("groundstate", H5T_NATIVE_DOUBLE, para.output_steps, para.nt);
    file.addDataset("rho00", H5T_NATIVE_DOUBLE, para.output_steps, para.nt);
    file.addDataset("rho11", H5T_NATIVE_DOUBLE, para.output_steps, para.nt);
    file.addDataset("rho22", H5T_NATIVE_DOUBLE, para.output_steps, para.nt);
    file.addDataset("rho33", H5T_NATIVE_DOUBLE, para.output_steps, para.nt);
    file.addAttribute("dt", para.dt);
    file.addAttribute("omega_svea", para.omega_svea);
    file.addAttribute("rfocus", pulse.rfocus);

    Propagator propagator(NSTATES);
    Matrix H(NSTATES, NSTATES);
    atom.rho.resize(NSTATES, NSTATES);

    vector<double> xfel(para.nt);
    vector<dcomplex> field(para.nt, 0.0);
    vector<dcomplex> pol(para.nt);
    vector<dcomplex> old_pol(para.nt);

    // calculate beam waist for Gaussian beam
    ofstream out("focus.dat");
    vector<double> beam_waist(para.nz+1);
    for (int i=0; i<para.nz+1; i++) {
        double z = i*para.dz - 0.5*para.propagation_length;
        beam_waist[i] = pulse.rfocus*gaussian_beam_waist(z, pulse.rayleigh_length);
        out<<z<<"  "<<beam_waist[i]<<endl;
    }
    out.close();

    vector<double> population[5];
    for (int i=0; i<5; i++) {
        population[i].resize(para.nt);
    }

    // init gaussian random numbers with random seed
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist;

    generate_pulse(pulse, para, gen, xfel);

    // renormalize initial XFEL flux because of beam waist
    for (int i=0; i<para.nt; i++) {
        xfel[i] *= pulse.rfocus*pulse.rfocus / (beam_waist[0]*beam_waist[0]);
    }

    // solid angle for r=200nm and L=200um
    double angle = M_PI*1.0e-6;

    double amplitude[NSTATES-1];
    double alpha = 2.0*M_PI*para.omega_svea;
    for (int i=0; i<NSTATES-1; i++) {
        double beta = 0.5*(atom.auger1s + atom.auger2p + atom.gamma_rad[i]);
        amplitude[i] = 8.0*M_PI*atom.gamma_rad[i]*atom.omega[i]*angle*beta*beta /
                       (alpha*alpha*atom.dipole[i]*atom.dipole[i]);
    }

    for (int i=0; i<para.nz; i++) {
        // boundary condition for density matrix
        atom.rho.setZeros();
        atom.groundstate = 1.0;
        double area_old = beam_waist[i]*beam_waist[i];
        double area_new = beam_waist[i+1]*beam_waist[i+1];
        // solve for density matrix
        for (int j=0; j<para.nt; j++) {
            double t = j*para.dt;
            // store density matrix value
            population[0][j] = atom.groundstate;
            population[1][j] = real(atom.rho(0, 0));
            population[2][j] = real(atom.rho(5, 5));
            population[3][j] = real(atom.rho(6, 6));
            population[4][j] = real(atom.rho(11, 11));

            // random source term for spontaneous emission
            dcomplex source[NSTATES-1];
            double pop = max(real(atom.rho(0, 0)), 0.0);
            for (int k=0; k<NSTATES-1; k++) {
                source[k] = dcomplex(dist(gen), dist(gen)) / sqrt(2.0) * sqrt(amplitude[k]*pop*para.dt);
            }

            // first decay step
            decayStep(atom, xfel[j], source, para.dt);

            H.setZeros();

            for (int k=0; k<NSTATES-1; k++) {
                H(k+1, 0) = -0.5*atom.dipole[k]*conj(field[j])*exp(dcomplex(0.0, (para.omega_svea - atom.omega[k])*t));
            }

            propagator.propagate(para.dt, atom.rho, H);

            // second decay step
            decayStep(atom, xfel[j], source, para.dt);

            // polarization
            dcomplex sum = 0.0;
            for (int k=0; k<NSTATES-1; k++) {
                sum += atom.dipole[k]*conj(atom.rho(k+1, 0))*exp(dcomplex(0.0, (para.omega_svea - atom.omega[k])*t));
            }

            double alpha = para.density*atom.reabsorption;

            pol[j] = dcomplex(0.0, 1.0)*4.0*M_PI*para.omega_svea/C*para.density*sum - alpha*field[j];

            // absorption of xfel pulse
            xfel[j] -= xfel[j]*(para.density*atom.sigma1s*atom.groundstate + atom.attenuation_solution)*para.dz;
            // renormalize XFEL flux because of change in beam waist area
            xfel[j] *= area_old / area_new;
        }

         // solve for field
        // for the first step only euler
        if (i == 0) {
            for (int j=0; j<para.nt; j++) {
                field[j] += para.dz*pol[j];
                old_pol[j] = pol[j];
            }
        }
        // 2nd order Adams-Bashforth
        else {
            for (int j=0; j<para.nt; j++) {
                field[j] += 1.5*para.dz*pol[j] - 0.5*para.dz*old_pol[j];
                old_pol[j] = pol[j];
            }
        }
        // write output to disk
        if (i % step == 0) {
            int row = i / step;
            cout<<"Row:  "<<row<<"  of  "<<para.output_steps<<endl;
            file.writeRow("xfel", row, xfel.data());
            file.writeRow("field", row, field.data());
            file.writeRow("groundstate", row, population[0].data());
            file.writeRow("rho00", row, population[1].data());
            file.writeRow("rho11", row, population[2].data());
            file.writeRow("rho22", row, population[3].data());
            file.writeRow("rho33", row, population[4].data());
        }
    }
}
