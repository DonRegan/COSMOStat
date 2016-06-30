#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <complex>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <gsl/gsl_sf_bessel.h>
//#include <omp.h>
#include "util.h"
#include "omp_cosmoStat.h"

using namespace std;


/* +++++++++++++++++++++++++++++++++ CONSTRUCTORS +++++++++++++++++++++++++++++++++++*/

COSMOStat::COSMOStat ()
{
}


COSMOStat::COSMOStat (int num_dimension, int num_grid, double boxsize)
{
  if (num_dimension != 2 && num_dimension != 3)
  {
    cout << "ERROR. The supported number of dimensions are either 2 or 3." << endl;
    exit(EXIT_FAILURE);
  }

  dim_ = num_dimension;
  n_ = num_grid;
  fn_ = n_/2+1;
  ndim_ = pow(num_grid,num_dimension);
  fndim_ = pow(num_grid,num_dimension-1)*(int(n_/2)+1);
  l_ = boxsize;
  kf_ = 2*M_PI/l_;

  util_.set_grid(dim_, n_);

  rho_ = new double[ndim_];
  frho_ = new fftw_complex[fndim_];
  frho2_ = new double[fndim_];
  idk_ = new vector<int>[dim_];

  if (dim_ == 2)
  {
    #pragma omp critical (make_plan)
    {
      p_rho_ = fftw_plan_dft_r2c_2d(n_, n_, rho_, frho_, FFTW_MEASURE);
      ip_rho_ = fftw_plan_dft_c2r_2d(n_, n_, frho_, rho_, FFTW_MEASURE);
    }
  }
  else
  {
    #pragma omp critical (make_plan)
    {
      p_rho_ = fftw_plan_dft_r2c_3d(n_, n_, n_, rho_, frho_, FFTW_MEASURE);
      ip_rho_ = fftw_plan_dft_c2r_3d(n_, n_, n_, frho_, rho_, FFTW_MEASURE);
    }
  }

  for (int ii=0; ii<fndim_; ii++)
  {
    double k2 = 0.;
    for (int d=0; d<dim_; d++)
    {
      idk_[d].push_back(util_.i_to_m(util_.fCoordId(ii,d)));
      k2 += pow(idk_[d][ii],2);
    }
    absk_.push_back(sqrt(k2)*kf_);
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    rho_[ii] = 0.;
  }
}


COSMOStat::COSMOStat (int num_dimension, int num_grid, double boxsize, long init_seed)
{
  if (num_dimension != 2 && num_dimension != 3)
  {
    cout << "ERROR. The supported number of dimensions are either 2 or 3." << endl;
    exit(EXIT_FAILURE);
  }

  dim_ = num_dimension;
  n_ = num_grid;
  fn_ = n_/2+1;
  ndim_ = pow(num_grid,num_dimension);
  fndim_ = pow(num_grid,num_dimension-1)*(int(n_/2)+1);
  l_ = boxsize;
  kf_ = 2*M_PI/l_;

  util_.set_grid(dim_, n_);

  rho_ = new double[ndim_];
  frho_ = new fftw_complex[fndim_];
  frho2_ = new double[fndim_];
  idk_ = new vector<int>[dim_];

  if (dim_ == 2)
  {
    #pragma omp critical (make_plan)
    {
      p_rho_ = fftw_plan_dft_r2c_2d(n_, n_, rho_, frho_, FFTW_MEASURE);
      ip_rho_ = fftw_plan_dft_c2r_2d(n_, n_, frho_, rho_, FFTW_MEASURE);
    }
  }
  else
  {
    #pragma omp critical (make_plan)
    {
      p_rho_ = fftw_plan_dft_r2c_3d(n_, n_, n_, rho_, frho_, FFTW_MEASURE);
      ip_rho_ = fftw_plan_dft_c2r_3d(n_, n_, n_, frho_, rho_, FFTW_MEASURE);
    }
  }

  for (int ii=0; ii<fndim_; ii++)
  {
    double k2 = 0.;
    for (int d=0; d<dim_; d++)
    {
      idk_[d].push_back(util_.i_to_m(util_.fCoordId(ii,d)));
      k2 += pow(idk_[d][ii],2);
    }
    absk_.push_back(sqrt(k2)*kf_);
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    rho_[ii] = 0.;
  }

  seed_ = init_seed;
  ran2(&seed_);
}


COSMOStat::~COSMOStat ()
{
  delete [] rho_;
  delete [] frho_;
  delete [] frho2_;

  fftw_destroy_plan(p_rho_);
  fftw_destroy_plan(ip_rho_);
}


/* +++++++++++++++++++++++++++++++++ I/O-ROUTINES +++++++++++++++++++++++++++++++++++*/

void COSMOStat::load (string fname)
{
  double buffer;
  int ii = 0;

  fstream data;
  data.open(fname.c_str(), ios::in);

  if (!data)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  while (data >> buffer)
  {
    if (ii == ndim_)
    {
      cout << "ERROR. Wrong file format." << endl;
      exit(EXIT_FAILURE);
    }

    rho_[ii] = buffer;
    ii++;
  }

  data.close();

  fftw_execute(p_rho_);
  util_.fftw_normalize(frho_);
}


void COSMOStat::load_particles (string loc, int nreal, double subfrac)
{
  char path[200], input_fname[200], basename[200];
  int snapshot_number, NumPart;
  long seed = -7348716*nreal;

  // Number of files per snapshot
  int files = 10;

  if (nreal < 8)
  {
    snapshot_number = 16;
  }
  else if (nreal < 30)
  {
    snapshot_number = 7;
  }
  else if (nreal < 40)
  {
    snapshot_number = 2;
  }
  else
  {
    snapshot_number = 3;
  }

  // Get input filenames
  stringstream fend, snap;
  fend << "LCDM-750-run" << nreal+1;
  snap << "snap" << snapshot_number;
  sprintf(path, (loc+fend.str()+"/DATA").c_str());
  sprintf(basename, "LCDM-L1500-N750-Tf_om_m_0.25_om_de_0.75_om_b_0.04_sig8_0.8_h_0.7");
  sprintf(input_fname, "%s/%s_%03d", path, basename, snapshot_number);

  // Load particle data into "P" and convert into smooth density field using CIC
  for (int i=0; i<files; i++)
  {
    particle_data_pos *P = load_sub_snapshot(input_fname, i, NumPart);
    if (subfrac < 1)
    {
      particle_data_pos *subP = subsample(P, NumPart, subfrac, seed);
      cic(subP, int(NumPart*subfrac));
      delete [] subP;
    }
    else
    {
      cic(P, NumPart);
    }
    delete [] P;
  }

  rho2delta();
  do_FFT();
}


void COSMOStat::cic (particle_data_pos *P, int NumPart)
{
  double spacing = l_/n_;
  for (unsigned int p=0; p<NumPart; p++)
  {
    int i = floor(P[p].Pos[0]/spacing);
    int j = floor(P[p].Pos[1]/spacing);
    int k = floor(P[p].Pos[2]/spacing);

    double dx = P[p].Pos[0]/spacing-i;
    double dy = P[p].Pos[1]/spacing-j;
    double dz = P[p].Pos[2]/spacing-k;
    double tx = 1-dx;
    double ty = 1-dy;
    double tz = 1-dz;

    int ipp = (i+1)%n_;
    int jpp = (j+1)%n_;
    int kpp = (k+1)%n_;

    rho_[util_.VecId(i,j,k)] += tx*ty*tz;
    rho_[util_.VecId(ipp,j,k)] += dx*ty*tz;
    rho_[util_.VecId(i,jpp,k)] += tx*dy*tz;
    rho_[util_.VecId(i,j,kpp)] += tx*ty*dz;
    rho_[util_.VecId(ipp,jpp,k)] += dx*dy*tz;
    rho_[util_.VecId(ipp,j,kpp)] += dx*ty*dz;
    rho_[util_.VecId(i,jpp,kpp)] += tx*dy*dz;
    rho_[util_.VecId(ipp,jpp,kpp)] += dx*dy*dz;
  }
}


struct particle_data_pos* COSMOStat::subsample (particle_data_pos *P, int NumPart,
  double fraction, long seed)
{
  int subNumPart = int(fraction*NumPart);
  struct particle_data_pos *subP = new particle_data_pos[subNumPart];

  ran2(&seed);

  for (int i=0; i<subNumPart; i++)
  {
    int p = int(ran2(&seed)*NumPart);
    subP[i].Pos[0] = P[p].Pos[0];
    subP[i].Pos[1] = P[p].Pos[1];
    subP[i].Pos[2] = P[p].Pos[2];
  }
  return subP;
}


void COSMOStat::rho2delta ()
{
  double rhoAvg = 0.;
  for (int ii=0; ii<ndim_; ii++)
  {
    rhoAvg += rho_[ii];
  }
  rhoAvg /= ndim_;
  for (int ii=0; ii<ndim_; ii++)
  {
    rho_[ii] = rho_[ii]/rhoAvg-1;
  }
}


void COSMOStat::save (string fname)
{
  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  if (dim_ == 2)
  {
    for (int i=0; i<n_; i++)
    {
      for (int j=0; j<n_; j++)
      {
        out << rho_[util_.VecId(i,j)] << "\t";
      }
      out << endl;
    }
  }
  else
  {
    for (int ii=0; ii<ndim_; ii++)
    {
      out << rho_[ii] << endl;
    }
  }

  out.close();
}


void COSMOStat::save_slice(string fname, int d, int val)
{
  int ii;
  fstream out;
  out.open(fname.c_str(), ios::out);

  for (int i=0; i<n_; i++)
  {
    for (int j=0; j<n_; j++)
    {
      switch (d)
      {
        case 1:
        ii = util_.VecId(val, i, j);
        break;
        case 2:
        ii = util_.VecId(i, val, j);
        break;
        case 3:
        ii = util_.VecId(i, j, val);
        break;
        default:
        cout << "Error. Invalid dimension - "
        << "choose either 1 (x), 2 (y) or 3 (z)." << endl;
        exit(EXIT_FAILURE);
      }
      out << rho_[ii] << "\t";
    }
    out << endl;
  }

  out.close();
}


void COSMOStat::set_Rho (double val, int id)
{
  rho_[id] = val;
}


void COSMOStat::set_RhoSubCube (double *parentRho, int parentN, int subId)
  {
    int x[dim_], subx[dim_];
    UTIL parentGrid(dim_, parentN), subGrid(dim_, parentN/n_);

    for (int d=0; d<dim_; d++)
    {
      subx[d] = subGrid.CoordId(subId,d)*n_;
    }
    for (int ii=0; ii<ndim_; ii++)
    {
      for (int d=0; d<dim_; d++)
      {
        x[d] = util_.CoordId(ii,d)+subx[d];
      }
      rho_[ii] = parentRho[parentGrid.VecId(x)];
    }

    do_FFT();

    for (int ii=0; ii<fndim_; ii++)
    {
      double window = 1.;
      for (int d=0; d<dim_; d++)
      {
        window *= sinc(M_PI*idk_[d][ii]/n_);
      }
      double re = frho_[ii][0];
      double im = frho_[ii][1];
      frho2_[ii] = (re*re+im*im)/pow(window,4);
    }
  }


/* +++++++++++++++++++++++++++++++++ SUBROUTINES ++++++++++++++++++++++++++++++++++++*/

void COSMOStat::shift (fftw_complex *field, double *dx)
{
  fftw_complex *ffield = new fftw_complex[ndim_];
  fftw_plan ip;

  if (dim_ == 2)
  {
    ip = fftw_plan_dft_2d(n_, n_, ffield, field, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else
  {
    ip = fftw_plan_dft_3d(n_, n_, n_, ffield, field, FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    double phase = 0.;
    for (int d=0; d<dim_; d++)
    {
      phase += idk_[d][ii]*dx[d];
    }
    phase *= 2*M_PI;
    double CosPhase = cos(phase);
    double SinPhase = sin(phase);
    double re = frho_[ii][0]*CosPhase-frho_[ii][1]*SinPhase;
    double im = frho_[ii][0]*SinPhase+frho_[ii][1]*CosPhase;
    ffield[ii][0] = re;
    ffield[ii][1] = im;
  }

  fftw_execute(ip);

  fftw_free(ffield);
  fftw_destroy_plan(ip);
}


void COSMOStat::shift (double *field_pos, double *field_neg, double *dx)
{
  fftw_complex *ffield_pos = new fftw_complex[fndim_];
  fftw_complex *ffield_neg = new fftw_complex[fndim_];
  fftw_plan ip_pos, ip_neg;

  if (dim_ == 2)
  {
    #pragma omp critical (make_plan)
    {
      ip_pos = fftw_plan_dft_c2r_2d(n_, n_, ffield_pos, field_pos, FFTW_ESTIMATE);
      ip_neg = fftw_plan_dft_c2r_2d(n_, n_, ffield_neg, field_neg, FFTW_ESTIMATE);
    }
  }
  else
  {
    #pragma omp critical (make_plan)
    {
      ip_pos = fftw_plan_dft_c2r_3d(n_, n_, n_, ffield_pos, field_pos, FFTW_ESTIMATE);
      ip_neg = fftw_plan_dft_c2r_3d(n_, n_, n_, ffield_neg, field_neg, FFTW_ESTIMATE);
    }
  }

  for (int ii=0; ii<fndim_; ii++)
  {
    double phase = 0.;
    for (int d=0; d<dim_; d++)
    {
      phase += idk_[d][ii]*dx[d];
    }
    phase *= 2*M_PI;
    double CosPhase = cos(phase);
    double SinPhase = sin(phase);
    double re1 = frho_[ii][0]*CosPhase;
    double re2 = frho_[ii][1]*SinPhase;
    double im1 = frho_[ii][0]*SinPhase;
    double im2 = frho_[ii][1]*CosPhase;
    ffield_pos[ii][0] = re1-re2;
    ffield_pos[ii][1] = im1+im2;
    ffield_neg[ii][0] = re1+re2;
    ffield_neg[ii][1] = -im1+im2;
  }

  fftw_execute(ip_pos);
  fftw_execute(ip_neg);

  fftw_free(ffield_pos);
  fftw_free(ffield_neg);
  fftw_destroy_plan(ip_pos);
  fftw_destroy_plan(ip_neg);
}


void COSMOStat::whiten (double threshold)
{
  for (int ii=0; ii<fndim_; ii++)
  {
    double re = frho_[ii][0], im = frho_[ii][1];
    double mag = sqrt(re*re+im*im), k2 = 0.;
    for (int d=0; d<dim_; d++)
    {
      k2 += pow(idk_[d][ii],2);
    }
    if (ii == 0 || mag < threshold)
    {
      frho_[ii][0] = 0.;
      frho_[ii][1] = 0.;
    }
    else
    {
      frho_[ii][0] *= exp(-pow(0*2*M_PI/n_,2)/2*k2)/mag;
      frho_[ii][1] *= exp(-pow(0*2*M_PI/n_,2)/2*k2)/mag;
    }
  }
  fftw_execute(ip_rho_);
  fftw_execute(p_rho_);
  util_.fftw_normalize(frho_);
}


void COSMOStat::filter (double scale, short filterMode)
{
  switch (filterMode)
  {
    case 1: // Top hat spherical cutoff
    for (int ii=0; ii<fndim_; ii++)
    {
      double k2 = 0.;
      for (int d=0; d<dim_; d++)
      {
        k2 += pow(idk_[d][ii],2);
      }
      double cutoff_scale = pow(scale/l_,2)*k2;
      if (cutoff_scale > 1.)
      {
        frho_[ii][0] = 0.;
        frho_[ii][1] = 0.;
      }
    }
    break;
    default:
    cout << "Error. Cutoff mode not defined." << endl;
    exit(EXIT_FAILURE);
  }
  fftw_execute(ip_rho_);
  fftw_execute(p_rho_);
  util_.fftw_normalize(frho_);
}


void COSMOStat::shell_c2r (double *rho_shell, double scale, double binSize)
{
  fftw_complex *frho_shell = new fftw_complex[fndim_];
  fftw_plan ip_rho_shell;

  if (dim_ == 2)
  {
    #pragma omp critical (make_plan)
    {
      ip_rho_shell = fftw_plan_dft_c2r_2d(n_, n_, frho_shell, rho_shell,
        FFTW_ESTIMATE);
    }
  }
  else
  {
    #pragma omp critical (make_plan)
    {
      ip_rho_shell = fftw_plan_dft_c2r_3d(n_, n_, n_, frho_shell, rho_shell,
        FFTW_ESTIMATE);
    }
  }

  for (int ii=0; ii<fndim_; ii++)
  {
    if (absk_[ii] > scale-binSize/2 && absk_[ii] < scale+binSize/2)
    {
      double window = 1.;
      for (int d=0; d<dim_; d++)
      {
        window *= sinc(M_PI*idk_[d][ii]/n_);
      }
      frho_shell[ii][0] = frho_[ii][0]/pow(window,2);
      frho_shell[ii][1] = frho_[ii][1]/pow(window,2);
    }
    else
    {
      frho_shell[ii][0] = 0.;
      frho_shell[ii][1] = 0.;
    }
  }

  fftw_execute(ip_rho_shell);

  fftw_free(frho_shell);
  fftw_destroy_plan(ip_rho_shell);
}


double COSMOStat::FieldInterpolation (double *x)
{
  double spacing = l_/n_;
  double avg = 0.;

  int i = floor(x[0]/spacing);
  int j = floor(x[1]/spacing);
  int k = floor(x[2]/spacing);

  double dx = x[0]/spacing-i;
  double dy = x[1]/spacing-j;
  double dz = x[2]/spacing-k;
  double tx = 1-dx;
  double ty = 1-dy;
  double tz = 1-dz;

  int ipp = (i+1)%n_;
  int jpp = (j+1)%n_;
  int kpp = (k+1)%n_;

  avg += rho_[util_.VecId(i,j,k)] * tx*ty*tz;
  avg += rho_[util_.VecId(ipp,j,k)] * dx*ty*tz;
  avg += rho_[util_.VecId(i,jpp,k)] * tx*dy*tz;
  avg += rho_[util_.VecId(i,j,kpp)] * tx*ty*dz;
  avg += rho_[util_.VecId(ipp,jpp,k)] * dx*dy*tz;
  avg += rho_[util_.VecId(ipp,j,kpp)] * dx*ty*dz;
  avg += rho_[util_.VecId(i,jpp,kpp)] * tx*dy*dz;
  avg += rho_[util_.VecId(ipp,jpp,kpp)] * dx*dy*dz;

  return avg;
}


vector<double> COSMOStat::get_nTriangle (double kmin, double kmax, double dk,
  double k2_rel, double k3_rel)
{
  double scale = kmin, k[3];

  double **shell = new double*[3];
  fftw_complex **fshell = new fftw_complex*[3];
  fftw_plan ip_shell[3];

  vector<double> nTriangle;

  for (int i=0; i<3; i++)
  {
    shell[i] = new double[ndim_];
    fshell[i] = new fftw_complex[fndim_];
  }

  if (dim_ == 2)
  {
    for (int i=0; i<3; i++)
    {
      ip_shell[i] = fftw_plan_dft_c2r_2d(n_, n_, fshell[i], shell[i],
        FFTW_ESTIMATE);
    }
  }
  else
  {
    for (int i=0; i<3; i++)
    {
      ip_shell[i] = fftw_plan_dft_c2r_3d(n_, n_, n_, fshell[i], shell[i],
        FFTW_ESTIMATE);
    }
  }

  while (scale < kmax)
  {
    k[0] = scale;
    k[1] = k2_rel*scale;
    k[2] = k3_rel*scale;

    for (int ii=0; ii<fndim_; ii++)
    {
      for (int i=0; i<3; i++)
      {
        if (absk_[ii] > k[i]-dk/2 && absk_[ii] < k[i]+dk/2)
        {
          fshell[i][ii][0] = 1.;
          fshell[i][ii][1] = 1.;
        }
        else
        {
          fshell[i][ii][0] = 0.;
          fshell[i][ii][1] = 0.;
        }
      }
    }

    for (int i=0; i<3; i++)
    {
      fftw_execute(ip_shell[i]);
    }

    double ntr = 0.;
    for (int ii=0; ii<ndim_; ii++)
    {
      ntr += shell[0][ii]*shell[1][ii]*shell[2][ii];
    }

    nTriangle.push_back(ntr/ndim_);
    scale += dk;
  }

  for (int i=0; i<3; i++)
  {
    delete [] shell[i];
    delete [] fshell[i];
    fftw_destroy_plan(ip_shell[i]);
  }

  return nTriangle;
}


vector<double> COSMOStat::get_nTriangle (double kmin, double kmax, double dk)
{
  double k[3];
  for (int i=0; i<3; i++) k[i] = kmin;

  double **shell = new double*[3];
  fftw_complex **fshell = new fftw_complex*[3];
  fftw_plan ip_shell[3];

  vector<double> nTriangle;

  for (int i=0; i<3; i++)
  {
    shell[i] = new double[ndim_];
    fshell[i] = new fftw_complex[fndim_];
  }

  if (dim_ == 2)
  {
    for (int i=0; i<3; i++)
    {
      ip_shell[i] = fftw_plan_dft_c2r_2d(n_, n_, fshell[i], shell[i],
        FFTW_ESTIMATE);
    }
  }
  else
  {
    for (int i=0; i<3; i++)
    {
      ip_shell[i] = fftw_plan_dft_c2r_3d(n_, n_, n_, fshell[i], shell[i],
        FFTW_ESTIMATE);
    }
  }

  while (k[0] < kmax)
  {
    k[1] = k[0];
    while (k[1] < kmax)
    {
      k[2] = k[1];
      while (k[2] < kmax)
      {
        if (k[2] < k[0]+k[1])
        {
          for (int ii=0; ii<fndim_; ii++)
          {
            for (int i=0; i<3; i++)
            {
              if (absk_[ii] > k[i]-dk/2 && absk_[ii] < k[i]+dk/2)
              {
                fshell[i][ii][0] = 1.;
                fshell[i][ii][1] = 0.;
              }
              else
              {
                fshell[i][ii][0] = 0.;
                fshell[i][ii][1] = 0.;
              }
            }
          }

          for (int i=0; i<3; i++) fftw_execute(ip_shell[i]);

          double ntr = 0.;
          for (int ii=0; ii<ndim_; ii++)
          {
            ntr += shell[0][ii]*shell[1][ii]*shell[2][ii];
          }

          nTriangle.push_back(ntr/ndim_);
        }
        k[2] += dk;
      }
      k[1] += dk;
    }
    k[0] += dk;
  }

  for (int i=0; i<3; i++)
  {
    delete [] shell[i];
    delete [] fshell[i];
    fftw_destroy_plan(ip_shell[i]);
  }

  return nTriangle;
}


void COSMOStat::id_mod (vector<idpair> *idmod)
{
  // for (int ii=0; ii<fndim_; ii++)
  // {
  //   idpair buffer;
  //   buffer.first = absk_[ii];
  //   buffer.second = ii;
  //   (*idmod).push_back(buffer);
  // }
  for (int ii=0; ii<ndim_; ii++)
  {
    double k2 =0.;
    idpair buffer;
    for (int d=0; d<dim_; d++)
    {
      k2 += pow(util_.i_to_m(util_.CoordId(ii,d)),2);
    }
    buffer.first = sqrt(k2)*kf_;
    buffer.second = ii;
    (*idmod).push_back(buffer);
  }
  sort((*idmod).begin(), (*idmod).end(), comparator);
}


double COSMOStat::get_RhoAvg ()
{
  double rhoAvg = 0.;
  for (int ii=0; ii<ndim_; ii++)
  {
    rhoAvg += rho_[ii];
  }
  return rhoAvg/ndim_;
}


double COSMOStat::get_PowerSpec (double k, double dk)
{
  double power = 0.;
  int nk = 0;
  for (int ii=0; ii<fndim_; ii++)
  {
    if (absk_[ii] > k-dk/2 && absk_[ii] < k+dk/2)
    {
      power += frho2_[ii];
      nk++;
    }
  }
  power *= pow(l_,dim_)/nk;

  return power;
}


void COSMOStat::generate_NGfield (int p)
{
  fftw_complex *ngfield = new fftw_complex[ndim_];
  fftw_complex *fngfield = new fftw_complex[ndim_];
  fftw_plan p_ng, ip_ng;

  if (dim_ == 2)
  {
    #pragma omp critical (make_plan)
    {
      p_ng = fftw_plan_dft_2d(n_, n_, ngfield, fngfield, FFTW_FORWARD, FFTW_ESTIMATE);
      ip_ng = fftw_plan_dft_2d(n_, n_, fngfield, ngfield, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  else
  {
    #pragma omp critical (make_plan)
    {
      p_ng = fftw_plan_dft_3d(n_, n_, n_, ngfield, fngfield, FFTW_FORWARD, FFTW_ESTIMATE);
      ip_ng = fftw_plan_dft_3d(n_, n_, n_, fngfield, ngfield, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  // int mu1 = 1, mu2 = 1;
  // for (int i=1; i<2*p; i+=2) mu1 *= i;
  // for (int i=1; i<4*p; i+=2) mu2 *= i;
  // mu2 -= mu1*mu1;

  double mu1 = 0.648721;
  double mu2 = 4.67077;
  double exponent = 4.25;
  double a = 1.0;

  for (int ii=0; ii<ndim_; ii++)
  {
    double u1 = ran2(&seed_);
    double u2 = ran2(&seed_);
    // ngfield[ii][0] = pow(sqrt(-2*log(u1))*cos(2*M_PI*u2), 2*p) - mu1;
    // ngfield[ii][0] = pow(abs(sqrt(-2*log(u1))*cos(2*M_PI*u2)), exponent) - mu1;
    ngfield[ii][0] = (exp(a*sqrt(-2*log(u1))*cos(2*M_PI*u2))-1.)/a - mu1;
    ngfield[ii][1] = 0.;
  }

  fftw_execute(p_ng);
  for (int ii=0; ii<ndim_; ii++)
  {
    fngfield[ii][0] /= ndim_;
    fngfield[ii][1] /= ndim_;
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    double k;
    if (dim_ == 2)
    k = 2*M_PI/l_*sqrt(pow(util_.i_to_m(util_.x(ii)),2) + pow(util_.i_to_m(util_.y(ii)),2));
    else
    k = 2*M_PI/l_*sqrt(pow(util_.i_to_m(util_.x(ii)),2) + pow(util_.i_to_m(util_.y(ii)),2) + pow(util_.i_to_m(util_.z(ii)),2));

    fngfield[ii][0] *= sqrt(k/pow(k+0.1,4)*ndim_/pow(l_,dim_)/mu2);
    fngfield[ii][1] *= sqrt(k/pow(k+0.1,4)*ndim_/pow(l_,dim_)/mu2);
  }

  fngfield[0][0] = 0.;
  fngfield[0][1] = 0.;

  fftw_execute(ip_ng);

  for (int ii=0; ii<ndim_; ii++)
  {
    rho_[ii] = ngfield[ii][0];
  }

  fftw_execute(p_rho_);
  util_.fftw_normalize(frho_);

  delete [] ngfield;
  delete [] fngfield;
  fftw_destroy_plan(p_ng);
  fftw_destroy_plan(ip_ng);
}


/* +++++++++++++++++++++++++++++++++ ESTIMATORS +++++++++++++++++++++++++++++++++++++*/

void COSMOStat::compute_LineCorr (string fname, double rmin, double rmax, double dr, short filterMode)
{
  double *rho_pos = new double[ndim_];
  double *rho_neg = new double[ndim_];

  double *r = new double[dim_];
  double scale = rmin, phi, theta, dphi, dtheta, sum, l;
  double kf2 = kf_*kf_;
  int nAngle, nAngle2, wphi, wtheta;

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  // Prepare quantities for Fourier space approach to LCF
  vector<idpair> idmod;
  vector<double> mod;
  vector<int> *idk_F = new vector<int>[dim_];

  id_mod(&idmod);
  for (int ii=0; ii<ndim_; ii++)
  {
    mod.push_back(idmod[ii].first);
  }
  for (int ii=0; ii<ndim_; ii++)
  {
    for (int d=0; d<dim_; d++)
    {
      idk_F[d].push_back(util_.i_to_m(util_.CoordId(ii,d)));
    }
  }

  fftw_complex *rho_F = new fftw_complex[ndim_];
  fftw_complex *frho_F = new fftw_complex[ndim_];
  fftw_plan p_rho_F, ip_rho_F;

  if (dim_ == 2)
  {
    #pragma omp critical (make_plan)
    {
      p_rho_F = fftw_plan_dft_2d(n_, n_, rho_F, frho_F, FFTW_FORWARD, FFTW_MEASURE);
      ip_rho_F = fftw_plan_dft_2d(n_, n_, frho_F, rho_F, FFTW_BACKWARD, FFTW_MEASURE);
    }
  }
  else
  {
    #pragma omp critical (make_plan)
    {
      p_rho_F = fftw_plan_dft_3d(n_, n_, n_, rho_F, frho_F, FFTW_FORWARD, FFTW_MEASURE);
      ip_rho_F = fftw_plan_dft_3d(n_, n_, n_, frho_F, rho_F, FFTW_BACKWARD, FFTW_MEASURE);
    }
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    rho_F[ii][0] = rho_[ii];
    rho_F[ii][1] = 0.;
  }

  fftw_execute(p_rho_F);
  for (int ii=0; ii<ndim_; ii++)
  {
    frho_F[ii][0] /= ndim_;
    frho_F[ii][1] /= ndim_;
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    double re = frho_F[ii][0], im = frho_F[ii][1];
    double mag = sqrt(re*re+im*im);
    if (ii == 0 || mag < 1e-7)
    {
      frho_F[ii][0] = 0.;
      frho_F[ii][1] = 0.;
    }
    else
    {
      frho_F[ii][0] /= mag;
      frho_F[ii][1] /= mag;
    }
  }

  whiten(1e-7);

  cout << "\t Scale [l_]" << "\t\t Line Correlation" << endl;
  cout << "----------------------------------------------------------" << endl;

  while (scale < rmax)
  {
    l = 0.;
    if (scale < 0.07*l_)
    {
      nAngle = (floor(scale*n_/l_/10.)+1)*20;
      nAngle2 = nAngle*nAngle;
      dphi = 2*M_PI/nAngle;
      dtheta = M_PI/nAngle;

      filter(scale, filterMode);

      if (dim_ == 2)
      {
        for (int i=0; i<nAngle+1; i++)
        {
          if (i == 0 || i == nAngle)
          {
            wphi = 1;
          }
          else
          {
            wphi = 2;
          }

          phi = i*dphi;
          r[0] = scale*cos(phi)/l_;
          r[1] = scale*sin(phi)/l_;

          shift(rho_pos, rho_neg, r);

          sum = 0.;
          for (int ii=0; ii<ndim_; ii++)
          {
            sum += rho_[ii]*rho_pos[ii]*rho_neg[ii];
          }
          l += wphi*sum;
        }
      }
      else
      {
        for (int i=0; i<nAngle+1; i++)
        {
          if (i == 0 || i == nAngle)
          {
            wphi = 1;
          }
          else
          {
            wphi = 2;
          }

          for (int j=0; j<nAngle+1; j++)
          {
            if (j == 0 || j == nAngle)
            {
              wtheta = 1;
            }
            else
            {
              wtheta = 2;
            }

            phi = i*dphi;
            theta = j*dtheta;
            r[0] = scale*sin(theta)*cos(phi)/l_;
            r[1] = scale*sin(theta)*sin(phi)/l_;
            r[2] = scale*cos(theta)/l_;

            shift(rho_pos, rho_neg, r);

            sum = 0.;
            for (int ii=0; ii<ndim_; ii++)
            {
              sum += rho_[ii]*rho_pos[ii]*rho_neg[ii];
            }
            l += wphi*wtheta*sin(theta)*sum;
          }
        }
      }

      if (dim_ == 2)
      {
        l *= pow(scale/l_,3./2*dim_)*dphi/(4*M_PI*ndim_);
      }
      else
      {
        l *= pow(scale/l_,3./2*dim_)*dphi*dtheta/(16*M_PI*ndim_);
      }
    }
    else
    {
      double invscale2 = pow(2*M_PI/scale, 2);
      int nmax = lower_bound(mod.begin(), mod.end(), sqrt(invscale2)) - mod.begin();
      for (int ii=0; ii<nmax; ii++)
      {
        for (int jj=0; jj<nmax; jj++)
        {
          int *Iij = new int[dim_];

          double k2 = mod[ii]*mod[ii];
          double q2 = mod[jj]*mod[jj];

          fftw_complex frho_q, frho_kq;
          double mu = 0.;

          for (int d=0; d<dim_; d++)
          {
            mu += idk_F[d][idmod[ii].second]*idk_F[d][idmod[jj].second];
          }
          mu *= kf2;

          if (mu < (invscale2-k2-q2)/2)
          {
            double kqminus = k2+q2-2*mu;
            if (kqminus < 0) kqminus = 0.;
            else kqminus = sqrt(kqminus);

            for (int d=0; d<dim_; d++)
            {
              Iij[d] = util_.m_to_i(-idk_F[d][idmod[ii].second]-idk_F[d][idmod[jj].second]);
            }

            if (dim_ == 2)
            {
              l += gsl_sf_bessel_J0(kqminus*scale)*prod3(frho_F[idmod[ii].second], frho_F[idmod[jj].second], frho_F[util_.VecId(Iij)]);
            }
            else
            {
              l += sinc(kqminus*scale)*prod3(frho_F[idmod[ii].second], frho_F[idmod[jj].second], frho_F[util_.VecId(Iij)]);
            }
          }
          delete [] Iij;
        }
      }

      l *= pow(scale/l_,3./2*dim_);
    }

    out << scale << "\t" << l << endl;
    cout << "\t" << fixed << scale << "\t\t" << fixed << l << endl;

    if (scale < 80)
    {
      scale += 5*dr;
    }
    else if (scale < 200)
    {
      scale += 10*dr;
    }
    else
    {
      scale += 25*dr;
    }

    // scale += dr;
  }


  cout << "----------------------------------------------------------" << endl;

  out.close();

  delete [] r;
  delete [] rho_pos;
  delete [] rho_neg;

  delete [] idk_F;
  delete [] rho_F;
  delete [] frho_F;

  fftw_destroy_plan(p_rho_F);
  fftw_destroy_plan(ip_rho_F);
}


void COSMOStat::compute_LineCorr_2 (string fname, double rmin, double rmax, double dr, short filterMode)
{
  double scale = rmin, rf = l_/n_;
  int Nr, binSize = 2;

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  vector<double> absR;
  vector<int> *R = new vector<int>[dim_];
  for (int ii=0; ii<ndim_; ii++)
  {
    double r2 = 0.;
    for (int d=0; d<dim_; d++)
    {
      R[d].push_back(util_.CoordId(ii,d));
      r2 += pow(util_.CoordId(ii,d),2);
    }
    absR.push_back(sqrt(r2)*rf);
  }

  whiten(1e-7);

  cout << "\t Scale [l_]" << "\t\t Line Correlation" << endl;
  cout << "----------------------------------------------------------" << endl;

  while (scale < rmax)
  {
    int *id_dist = new int[dim_];
    vector<int> id_shell;
    double id_scale = scale/rf, id_scale_p = scale/rf+0.5*binSize, id_scale_m = sqrt(2)/2*(scale/rf-0.5*binSize);
    double l = 0.;

    filter(scale, filterMode);

    Nr = 0;
    for (int ii=0; ii<fndim_; ii++)
    {
      int b1 = 1, b2 = 1;
      for (int d=0; d<dim_; d++)
      {
        id_dist[d] = util_.distPBC(R[d][ii]);
        // check if within outer and inner bounding boxes
        if (id_dist[d] > id_scale_p)
        {
          b1 = 0;
          break;
        }
        if (id_dist[d] < id_scale_m)
        {
          b2 *= 1;
        }
        else
        {
          b2 *= 0;
        }
      }
      if (b1 && !b2)
      {
        double dist = 0.;
        for (int d=0; d<dim_; d++)
        {
          dist += id_dist[d]*id_dist[d];
        }
        if (abs(sqrt(dist)-id_scale) < 0.5*binSize)
        {
          id_shell.push_back(ii);
          Nr++;
        }
      }
    }

    for (int ii=0; ii<ndim_; ii++)
    {
      double avg = 0.;
      for (int j=0; j<Nr; j++)
      {
        int ii_pos, ii_neg;
        if (dim_ == 2)
        {
          ii_pos = util_.VecId((R[0][id_shell[j]]+R[0][ii])%n_, (R[1][id_shell[j]]+R[1][ii])%n_);
          ii_neg = util_.VecId(((-R[0][id_shell[j]]+R[0][ii]+n_)%n_), ((-R[1][id_shell[j]]+R[1][ii]+n_)%n_));
        }
        else
        {
          ii_pos = util_.VecId((R[0][id_shell[j]]+R[0][ii])%n_, (R[1][id_shell[j]]+R[1][ii])%n_, (R[2][id_shell[j]]+R[2][ii])%n_);
          ii_neg = util_.VecId((-R[0][id_shell[j]]+R[0][ii]+n_)%n_, (-R[1][id_shell[j]]+R[1][ii]+n_)%n_, (-R[2][id_shell[j]]+R[2][ii]+n_)%n_);
        }
        avg += rho_[ii_pos]*rho_[ii_neg];
      }

      l += rho_[ii]*avg;
    }

    l *= pow(scale/l_,3./2*dim_)/ndim_/Nr;

    out << scale << "\t" << l << endl;
    cout << "\t" << fixed << scale << "\t\t" << fixed << l << endl;

    scale += dr;
  }

  cout << "----------------------------------------------------------" << endl;

  out.close();
}


void COSMOStat::compute_LineCorr_F (string fname, double rmin, double rmax, double dr)
{
  double scale = rmin, invscale2;
  double kf2 = kf_*kf_;
  vector<idpair> idmod;
  vector<double> mod;

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  id_mod(&idmod);
  for (int ii=0; ii<ndim_; ii++)
  {
    mod.push_back(idmod[ii].first);
  }

  vector<int> *idk = new vector<int>[dim_];
  for (int ii=0; ii<ndim_; ii++)
  {
    for (int d=0; d<dim_; d++)
    {
      idk[d].push_back(util_.i_to_m(util_.CoordId(ii,d)));
    }
  }

  // whiten(1e-7);

  fftw_complex *rho = new fftw_complex[ndim_];
  fftw_complex *frho = new fftw_complex[ndim_];

  fftw_plan p_rho, ip_rho;
  if (dim_ == 2)
  {
    p_rho = fftw_plan_dft_2d(n_, n_, rho, frho, FFTW_FORWARD, FFTW_MEASURE);
    ip_rho = fftw_plan_dft_2d(n_, n_, frho, rho, FFTW_BACKWARD, FFTW_MEASURE);
  }
  else
  {
    p_rho = fftw_plan_dft_3d(n_, n_, n_, rho, frho, FFTW_FORWARD, FFTW_MEASURE);
    ip_rho = fftw_plan_dft_3d(n_, n_, n_, frho, rho, FFTW_BACKWARD, FFTW_MEASURE);
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    rho[ii][0] = rho_[ii];
    rho[ii][1] = 0.;
  }

  fftw_execute(p_rho);
  for (int ii=0; ii<ndim_; ii++)
  {
    frho[ii][0] /= ndim_;
    frho[ii][1] /= ndim_;
  }

  for (int ii=0; ii<ndim_; ii++)
  {
    double re = frho[ii][0], im = frho[ii][1];
    double mag = sqrt(re*re+im*im);
    if (ii == 0 || mag < 1e-7)
    {
      frho[ii][0] = 0.;
      frho[ii][1] = 0.;
    }
    else
    {
      frho[ii][0] /= mag;
      frho[ii][1] /= mag;
    }
  }

  cout << "\t Scale [l_]" << "\t\t Line Correlation" << endl;
  cout << "----------------------------------------------------------" << endl;

  while (scale < rmax)
  {
    invscale2 = (2*M_PI/scale)*(2*M_PI/scale);
    int nmax = lower_bound(mod.begin(), mod.end(), sqrt(invscale2)) - mod.begin();
    double l = 0.;

    for (int ii=0; ii<nmax; ii++)
    {
      for (int jj=0; jj<nmax; jj++)
      {
        int *Iij = new int[dim_];

        double k2 = mod[ii]*mod[ii];
        double q2 = mod[jj]*mod[jj];

        fftw_complex frho_q, frho_kq;
        double mu = 0.;

        for (int d=0; d<dim_; d++)
        {
          mu += idk[d][idmod[ii].second]*idk[d][idmod[jj].second];
        }
        mu *= kf2;

        if (mu < (invscale2-k2-q2)/2)
        {
          double kqminus = k2+q2-2*mu;
          if (kqminus < 0)
          kqminus = 0.;
          else
          kqminus = sqrt(kqminus);

          for (int d=0; d<dim_; d++)
          {
            Iij[d] = util_.m_to_i(-idk[d][idmod[ii].second]+
              -idk[d][idmod[jj].second]);
          }

          if (dim_ == 2)
          {
            l += gsl_sf_bessel_J0(kqminus*scale)*prod3(frho[idmod[ii].second],
              frho[idmod[jj].second], frho[util_.VecId(Iij)]);
          }
          else
          {
            l += sinc(kqminus*scale)*prod3(frho[idmod[ii].second],
              frho[idmod[jj].second], frho[util_.VecId(Iij)]);
          }

        }

        delete [] Iij;
      }
    }

    l *= pow(scale/l_,3./2*dim_);

    out << scale << "\t" << l << endl;
    cout << "\t" << fixed << scale << "\t\t" << fixed << l << endl;

    if (scale < 80)
    {
      scale += 5*dr;
    }
    else if (scale < 150)
    {
      scale += 10*dr;
    }
    else
    {
      scale += 25*dr;
    }

    // scale += dr;
  }

  cout << "----------------------------------------------------------" << endl;

  out.close();

  delete [] rho;
  delete [] frho;
  delete [] idk;
  fftw_destroy_plan(p_rho);
  fftw_destroy_plan(ip_rho);
}


void COSMOStat::compute_LineCorr_MC (string fname, double rmin, double rmax, double dr)
{
  double scale = rmin, invscale, l;
  double theta, phi;
  double ctheta, cphi, stheta, sphi;
  double rho1, rho2, rho3;

  double *x = new double[dim_];
  double *xpr = new double[dim_];
  double *xmr = new double[dim_];


  int nMC = 500000000;

  long seed = -8762349;
  ran2(&seed);

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }


  whiten(1e-7);

  cout << "\t Scale [l_]" << "\t\t Line Correlation" << endl;
  cout << "----------------------------------------------------------" << endl;

  while (scale < rmax)
  {
    l = 0.;
    invscale = TWOPI/scale;

    filter(scale, 1);

    for (int n=0; n<nMC; n++)
    {

      for (int d=0; d<dim_; d++)
      {
        x[d] = l_*ran2(&seed);
      }

      theta = M_PI*ran2(&seed);
      phi = TWOPI*ran2(&seed);

      ctheta = cos(theta);
      cphi = cos(phi);
      stheta = sin(theta);
      sphi = sin(phi);

      if (dim_ == 2)
      {
        xpr[0] = x[0]+scale*cphi;
        xpr[1] = x[1]+scale*sphi;
        xmr[0] = x[0]-scale*cphi;
        xmr[1] = x[1]-scale*sphi;
      }
      else
      {
        xpr[0] = x[0]+scale*stheta*cphi;
        xpr[1] = x[1]+scale*stheta*sphi;
        xpr[2] = x[2]+scale*ctheta;
        xmr[0] = x[0]-scale*stheta*cphi;
        xmr[1] = x[1]-scale*stheta*sphi;
        xmr[2] = x[2]-scale*ctheta;
      }

      for (int d=0; d<dim_; d++)
      {
        if(xpr[d] < 0) xpr[d] = l_-xpr[d];
        if(xmr[d] < 0) xmr[d] = l_-xmr[d];
      }

      rho1 = FieldInterpolation(x);
      rho2 = FieldInterpolation(xpr);
      rho3 = FieldInterpolation(xmr);

      if (dim_ == 2)
      {
        l += rho1*rho2*rho3;
      }
      else
      {
        l += stheta*rho1*rho2*rho3;
      }
    }

    l *= pow(scale/l_,1.5*dim_)*pow(M_PI/2,dim_-2)/nMC;

    out << scale << "\t" << l << endl;
    cout << "\t" << fixed << scale << "\t\t" << fixed << l << endl;

    if (scale < 80)
    {
      scale += 5*dr;
    }
    else if (scale < 200)
    {
      scale += 10*dr;
    }
    else
    {
      scale += 25*dr;
    }
  }

  cout << "----------------------------------------------------------" << endl;

  out.close();
}


void COSMOStat::compute_PowerSpec (string fname, double kmin, double kmax, double dk)
{
  double scale = kmin;

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "Error. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  for (int ii=0; ii<fndim_; ii++)
  {
    double window = 1.;
    for (int d=0; d<dim_; d++)
    {
      window *= sinc(M_PI*idk_[d][ii]/n_);
    }
    double re = frho_[ii][0];
    double im = frho_[ii][1];
    frho2_[ii] = (re*re+im*im)/pow(window,2);
  }

  cout << "\t Scale [1/l_]" << "\t\t Power Spectrum" << endl;
  cout << "----------------------------------------------------------" << endl;

  while (scale < kmax)
  {
    double power = 0.;
    int nk = 0;
    for (int ii=0; ii<fndim_; ii++)
    {
      if (absk_[ii] > scale-dk/2 && absk_[ii] < scale+dk/2)
      {
        power += frho2_[ii];
        nk++;
      }
    }
    power *= pow(l_,dim_)/nk;

    out << scale << "\t" << power << endl;
    cout << "\t " << fixed << scale << "\t\t " << fixed << power << endl;

    scale += dk;
  }

  cout << "----------------------------------------------------------" << endl;

  out.close();
}


void COSMOStat::compute_PowerSpec_2 (string fname, double kmin, double kmax, double dk)
{
  double scale = kmin;
  double *rho_shell = new double[ndim_];

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  if (kmin < kf_)
  {
    cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/l_."
    << endl;
    kmin = kf_;
    scale = kmin;
  }
  if (kmax > n_*kf_/2)
  {
    cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = n_*Pi/l_."
    << endl;
    kmax = n_*kf_/2;
  }

  cout << "\t k [1/l_]" << "\t\t Power Spectrum [l_^dim_]" << endl;
  cout << "----------------------------------------------------------" << endl;

  while (scale < kmax)
  {
    shell_c2r(rho_shell, scale, dk);

    double power = 0.;
    for (int ii=0; ii<ndim_; ii++)
    {
      power += pow(rho_shell[ii],2);
    }

    if (dim_ == 2)
    {
      power *= pow(kf_,2)/(2*M_PI*scale*(2*kf_));
    }
    else
    {
      power *= pow(kf_,3)/(4*M_PI*pow(scale,2)*2*kf_);
    }

    power *= pow(l_,dim_)/ndim_;

    out << scale << "\t" << power << endl;
    cout << "\t " << fixed << scale << "\t\t " << fixed << power << endl;

    scale += dk;
  }

  cout << "----------------------------------------------------------" << endl;
}


void COSMOStat::compute_PositionDependentPowerSpec (string fname, double kmin, double kmax, double dk,
  int nCut, int subId)
{
  COSMOStat subBox(dim_, n_/nCut, l_/nCut);
  subBox.set_RhoSubCube(rho_, n_, subId);
  subBox.do_FFT();
  subBox.compute_PowerSpec(fname, kmin, kmax, dk);
}


void COSMOStat::compute_IntegratedBiSpec (string fname, double kmin, double kmax, double dk, int nCut)
{
  double scale = kmin, kf = 2*M_PI/(l_/nCut);
  COSMOStat subBox(dim_, n_/nCut, l_/nCut);

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "ERROR. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  if (kmin < kf)
  {
    cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/(l_/nCut)."
    << endl;
    kmin = kf;
    scale = kmin;
  }
  if (kmax > n_*kf/2)
  {
    cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = n_*Pi/l_."
    << endl;
    kmax = n_*kf/2;
  }

  cout << "\t k [1/l_]" << "\t\t Integrated Bispectrum [V^2]" << endl;
  cout << "----------------------------------------------------------" << endl;

  vector<double> iB;
  while (scale < kmax)
  {
    iB.push_back(0.0);
    scale += dk;
  }
  scale = kmin;

  for (int i=0; i<pow(nCut,dim_); i++)
  {
    subBox.set_RhoSubCube(rho_, n_, i);
    double rhoAvg = subBox.get_RhoAvg();

    for (int j=0; j<iB.size(); j++)
    {
      double power = subBox.get_PowerSpec(scale, dk);
      iB[j] += power*rhoAvg;
      scale += dk;
    }
    scale = kmin;
  }

  for (int i=0; i<iB.size(); i++)
  {
    iB[i] /= pow(nCut,dim_);
  }

  for (int i=0; i<iB.size(); i++)
  {
    out << scale << "\t" << iB[i] << endl;
    cout << "\t " << fixed << scale << "\t\t " << fixed << iB[i] << endl;
    scale += dk;
  }

  cout << "----------------------------------------------------------" << endl;

  out.close();
}


void COSMOStat::compute_BiSpec (string fname, double kmin, double kmax, double dk,
  double k2_rel, double k3_rel)
{
  double scale = kmin;
  double *rho_shell1 = new double[ndim_];
  double *rho_shell2 = new double[ndim_];
  double *rho_shell3 = new double[ndim_];

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "Error. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  if (kmin < kf_)
  {
    cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/l_."
    << endl;
    kmin = kf_;
    scale = kmin;
  }
  if (kmax > n_*kf_/2)
  {
    cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = n_*Pi/l_."
    << endl;
    kmax = n_*kf_/2;
  }

  if (nTriangle_.size() == 0)
  {
    COSMOStat TriMesh(dim_,(1+floor(3.*int(dk/kf_)*int(kmax/kf_)/10))*10, l_);
    nTriangle_ = TriMesh.get_nTriangle(kmin, kmax, dk, k2_rel, k3_rel);
  }

  cout << "\t k_1 [1/l_]" << "\t\t Bispectrum (k_2 = " << k2_rel << "*k_1, k_3 = "
  << k3_rel << "*k_1)" << endl;
  cout << "----------------------------------------------------------" << endl;

  int nr = 0;
  while (scale < kmax)
  {
    shell_c2r(rho_shell1, scale, dk);
    shell_c2r(rho_shell2, k2_rel*scale, dk);
    shell_c2r(rho_shell3, k3_rel*scale, dk);

    double B = 0.;
    for (int ii=0; ii<ndim_; ii++)
    {
      B += rho_shell1[ii]*rho_shell2[ii]*rho_shell3[ii];
    }

    if (dim_ == 2)
    {
      B *= pow(kf_,4)/(4*M_PI*k3_rel*scale*pow(dk,3));
      // B /= nTriangle_[nr];
    }
    else
    {
      // B *= pow(kf_,6)/(8*M_PI*M_PI*k2_rel*k3_rel*pow(scale*dk,3));
      B /= nTriangle_[nr];
    }

    B *= pow(l_,2*dim_)/ndim_;

    out << scale << "\t" << B << endl;
    cout << "\t " << fixed << scale << "\t\t " << fixed << B << endl;

    scale += dk;
    nr++;
  }

  cout << "----------------------------------------------------------" << endl;

  delete [] rho_shell1;
  delete [] rho_shell2;
  delete [] rho_shell3;
}

void COSMOStat::compute_BiSpec (string fname, double kmin, double kmax, double dk)
{
  double k1 = kmin, k2 = kmin, k3 = kmin;
  double *rho_shell1 = new double[ndim_];
  double *rho_shell2 = new double[ndim_];
  double *rho_shell3 = new double[ndim_];

  fstream out;
  out.open(fname.c_str(), ios::out);

  if (!out)
  {
    cout << "Error. Cannot open file." << endl;
    exit(EXIT_FAILURE);
  }

  if (kmin < kf_)
  {
    cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/l_."
    << endl;
    kmin = kf_;
    scale = kmin;
  }
  if (kmax > n_*kf_/2)
  {
    cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = n_*Pi/l_."
    << endl;
    kmax = n_*kf_/2;
  }

  if (nTriangle_.size() == 0)
  {
    COSMOStat TriMesh(dim_,(1+floor(3.*int(dk/kf_)*int(kmax/kf_)/10))*10, l_);
    nTriangle_ = TriMesh.get_nTriangle(kmin, kmax, dk);
  }

  cout << "\t k_1, k_2, k_3 [1/l_]" << "\t\t Bispectrum" << endl;
  cout << "----------------------------------------------------------" << endl;


  int nr = 0;
  while (k1 < kmax)
  {
    k2 = k1;
    while (k2 < kmax)
    {
      k3 = k2;
      while (k3 < kmax)
      {
        if (k3 < k1+k2)
        {
          shell_c2r(rho_shell1, k1, dk);
          shell_c2r(rho_shell2, k2, dk);
          shell_c2r(rho_shell3, k3, dk);

          double B = 0.;
          for (int ii=0; ii<ndim_; ii++)
          {
            B += rho_shell1[ii]*rho_shell2[ii]*rho_shell3[ii];
          }

          if (dim_ == 2)
          {
            // B *= pow(kf_,4)/(4*M_PI*k3_rel*scale*pow(dk,3));
            B /= nTriangle_[nr];
          }
          else
          {
            // B *= pow(kf_,6)/(8*M_PI*M_PI*k2_rel*k3_rel*pow(scale*dk,3));
            B /= nTriangle_[nr];
          }

          B *= pow(l_,2*dim_)/ndim_;

          out << k1 << "\t" << k2 << "\t" << k3 << "\t" << B << endl;
          cout << "\t " << fixed << k1 << "\t" << k2 << "\t" << k3 << "\t\t " << fixed << B << endl;

          nr++;
        }
        k3 += dk;
      }
      k2 += dk;
    }
    k1 += dk;
  }

  cout << "----------------------------------------------------------" << endl;

  delete [] rho_shell1;
  delete [] rho_shell2;
  delete [] rho_shell3;
}










/* ++++++++++++++++++++++++++ MISC ++++++++++++++++++++++++++++++++++++++++++++++*/

// power spectrum estimation via direct sampling. Not working yet!
// void COSMOStat::get_powerspec_avg (string fname, double kmin, double kmax, double dk)
// {
//   double *frho2_ = new double[ndim_];
//   double scale = kmin, kf = 2*M_PI/l_;
//   vector<double> k2;

//   fstream out;
//   out.open(fname.c_str(), ios::out);

//   if (!out)
//     {
//       cout << "Error. Cannot open file." << endl;
//       exit(EXIT_FAILURE);
//     }

//   for (int ii=0; ii<ndim_; ii++)
//     {
//       double re = frho_[ii][0];
//       double im = frho_[ii][1];
//       frho2_[ii] = sqrt(re*re+im*im);
//     }

//   for (int ii=0; ii<ndim_; ii++)
//     {
//       k2.push_back(0.0);
//       for (int d=0; d<DIM; d++)
//  {
//    k2[ii] += pow(idk_[d][ii],2);
//  }
//       k2[ii] *= pow(kf,2);
//     }

//   cout << "\t Scale [1/l_]" << "\t\t Power Spectrum" << endl;
//   cout << "----------------------------------------------------------" << endl;

//   while (scale < kmax)
//     {
//       double power = 0.;
//       int nk = 0;
//       for (int ii=0; ii<ndim_; ii++)
//  {
//    if (sqrt(k2[ii]) > scale-kf/2 && sqrt(k2[ii]) < scale+kf/2)
//      {
//        power += frho2_[ii];
//        nk++;
//      }
//  }
//       power /= nk;

//       out << scale << "\t" << power << endl;
//       cout << "\t " << fixed << scale << "\t\t " << fixed << power << endl;

//       scale += 2*M_PI/l_;
//     }

//   cout << "----------------------------------------------------------" << endl;
// }

// void COSMOStat::cicNeighbours (double *k, vector<int> *id_neighbour, vector<double> cic_weight)
// {
//   double krel;
//   int *id_center = new int[dim_];
//   int *id_pp = new int[dim_];
//   double *dx = new double[dim_];
//   double *tx = new double[dim_];
//   double *weight = new double[dim_];
//   int n = 0;
//
//   for (int d=0; d<dim_; d++)
//   {
//     krel = k[d]/kf_;
//     id_center[d] = util_.m_to_i(int(krel));
//     id_pp[d] = (id_center[d]+1)%n_;
//     dx[d] = krel-int(krel);
//     tx[d] = 1.-dx[d];
//   }
//
//   if (dim_ == 2)
//   {
//     for (int i=0; i<2; i++)
//     {
//       if (i == 0) weight[0] = tx[0];
//       else weight[0] = dx[0];
//
//       for (int j=0; j<2; j++)
//       {
//         if (i == 0) id_neighbour[n].push_back(id_center[0]);
//         else  id_neighbour[n].push_back(id_pp[0]);
//         if (j == 0)
//         {
//           weight[1] = tx[1];
//           id_neighbour[n].push_back(id_center[1]);
//         }
//         else
//         {
//           weight[1] = dx[1];
//           id_neighbour[n].push_back(id_pp[1]);
//         }
//         cic_weight.push_back(weight[0]*weight[1]);
//         n++;
//       }
//     }
//   }
//   else
//   {
//     for (int i=0; i<2; i++)
//     {
//       if (i == 0) weight[0] = tx[0];
//       else weight[0] = dx[0];
//
//       for (int j=0; j<2; j++)
//       {
//         if (j == 0) weight[1] = tx[1];
//         else weight[1] = dx[1];
//
//         for (int l=0; l<2; l++)
//         {
//           if (i == 0) id_neighbour[n].push_back(id_center[0]);
//           else  id_neighbour[n].push_back(id_pp[0]);
//           if (j == 0) id_neighbour[n].push_back(id_center[1]);
//           else  id_neighbour[n].push_back(id_pp[1]);
//           if (l == 0)
//           {
//             weight[2] = tx[2];
//             id_neighbour[n].push_back(id_center[2]);
//           }
//           else
//           {
//             weight[2] = dx[2];
//             id_neighbour[n].push_back(id_pp[2]);
//           }
//           cic_weight.push_back(weight[0]*weight[1]*weight[2]);
//           n++;
//         }
//       }
//     }
//   }
//
//   delete [] id_center;
//   delete [] id_pp;
//   delete [] dx;
//   delete [] tx;
//   delete [] weight;
// }
//
//
// void COSMOStat::FourierModeInterpolation (fftw_complex fk, double* k)
// {
//   vector<int> *id_neighbour = new vector<int>[int(pow(2,dim_))];
//   vector<double> cic_weight;
//   fk[0] = 0.;
//   fk[1] = 0.;
//   cicNeighbours(k, id_neighbour, cic_weight);
//
//   for (int i=0; i<cic_weight.size(); i++)
//   {
//     if (id_neighbour[i][0] < fn_)
//     {
//       fk[0] += cic_weight[i]*frho_[util_.fVecId(id_neighbour[i])][0];
//       fk[1] += cic_weight[i]*frho_[util_.fVecId(id_neighbour[i])][1];
//     }
//     else
//     {
//       for (int d=0; d<dim_; d++)
//       {
//         id_neighbour[i][d] = n_-id_neighbour[i][d];
//       }
//       fk[0] += cic_weight[i]*frho_[util_.fVecId(id_neighbour[i])][0];
//       fk[1] -= cic_weight[i]*frho_[util_.fVecId(id_neighbour[i])][1];
//     }
//   }
//
//   delete [] id_neighbour;
// }
