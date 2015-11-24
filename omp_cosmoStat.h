#ifndef COSMOSTAT_H
#define COSMOSTAT_H

#include <string>
#include <vector>
#include <fftw3.h>
#include "util.h"
#include "proto.h"


class COSMOStat{
  
 private:
  
  int DIM;
  int N, NDIM, FNDIM;
  double L;

  UTIL util;

  double *rho;
  fftw_complex *frho;
  double *frho2;

  fftw_plan p_rho, ip_rho;

  std::vector<int> *K;
  std::vector<double> k2;
  std::vector<double> nTriangle;

  void shift (fftw_complex*, double*);
  void shift (double*, double*, double*);
  void whiten (double);
  void filter (double, short);
  void shell_c2r (double*, double, double, std::vector<double>);
  void est_nTriangle (double, double, double, double, double);


 public:
  
  COSMOStat ();
  COSMOStat (int, int, double);
  ~COSMOStat ();

  void load (std::string);
  void cic (particle_data_pos*, int);
  void rho2delta ();
  void save (std::string);
  void save_slice (std::string, int, int);
  
  void do_FFT()
  {
    fftw_execute(p_rho);
    util.fftw_normalize(frho);
  }

  void set_Rho (double, int);
  void set_RhoSubCube (double*, int, int);

  double get_Rho (int ii) { return rho[ii]; }
  double get_RhoAvg ();
  double get_PowerSpec (double, double);

  void compute_LineCorr (std::string, double, double, double, short);
  void compute_LineCorr_2 (std::string, double, double, double, short);
  void compute_PowerSpec (std::string, double, double, double);
  void compute_PowerSpec_2 (std::string, double, double, double);
  void compute_PositionDependentPowerSpec (std::string, double, double, double, int, int);
  void compute_IntegratedBiSpec (std::string, double, double, double, int);
  void compute_BiSpec (std::string, double, double, double, double, double);
};

#endif
