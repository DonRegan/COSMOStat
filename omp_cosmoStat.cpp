#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <complex>
#include <vector>
#include <stdlib.h>
#include <algorithm>
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
        double kf = 2*M_PI/l_;
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
        fftw_plan p, ip;

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


vector<int> COSMOStat::get_nTriangle (double kmin, double kmax, double dk,
                                      double k2_rel, double k3_rel)
{
        double scale = kmin, k[3];

        double **shell = new double*[3];
        fftw_complex **fshell = new fftw_complex*[3];
        fftw_plan ip_shell[3];

        vector<int> nTriangle;

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

                nTriangle.push_back(int(ntr/ndim_));
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


void COSMOStat::id_mod (vector<idpair> idmod)
{
  for (int ii=0; ii<fndim_; ii++)
  {
    idpair buffer;
    buffer.first = absk_[ii];
    buffer.second = ii;
    idmod.push_back(buffer);
  }
  sort(idmod.begin(), idmod.end(), comparator);
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

  double mu1 = 1.;
  double mu2 = 2.;
  double exponent = 2.;

  for (int ii=0; ii<ndim_; ii++)
    {
      double u1 = ran2(&seed);
      double u2 = ran2(&seed);
      // ngfield[ii][0] = pow(sqrt(-2*log(u1))*cos(2*M_PI*u2), 2*p) - mu1;
      ngfield[ii][0] = pow(abs(sqrt(-2*log(u1))*cos(2*M_PI*u2)), exponent) - mu1;
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
        fftw_complex *result = new fftw_complex[ndim_];
        double *rho_pos = new double[ndim_];
        double *rho_neg = new double[ndim_];
        fftw_plan p_linecorr;

        double *r = new double[dim_];
        double scale = rmin, phi, theta, weight;
        int nAngle, nAngle2;

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "ERROR. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        if (dim_ == 2)
        {
#pragma omp critical (make_plan)
                {
                        p_linecorr = fftw_plan_dft_2d(n_, n_, result, result,
                                                      FFTW_FORWARD, FFTW_MEASURE);
                }
        }
        else
        {
#pragma omp critical (make_plan)
                {
                        p_linecorr = fftw_plan_dft_3d(n_, n_, n_, result, result,
                                                      FFTW_FORWARD, FFTW_MEASURE);
                }
        }

        whiten(1e-7);

        cout << "\t Scale [l_]" << "\t\t Line Correlation" << endl;
        cout << "----------------------------------------------------------" << endl;

        while (scale < rmax)
        {
                nAngle = (floor(scale*n_/l_/10.)+1)*20;
                nAngle2 = nAngle*nAngle;
                for (int ii=0; ii<ndim_; ii++)
                {
                        result[ii][0] = 0.;
                        result[ii][1] = 0.;
                }

                filter(scale, filterMode);

                if (dim_ == 2)
                {
                        for (int i=0; i<nAngle+1; i++)
                        {
                                if (i == 0 || i == nAngle)
                                {
                                        weight = 1./(2*nAngle);
                                }
                                else
                                {
                                        weight = 1./(nAngle);
                                }

                                phi = M_PI/nAngle*i;
                                r[0] = scale*cos(phi)/l_;
                                r[1] = scale*sin(phi)/l_;

                                shift(rho_pos, rho_neg, r);

                                for (int ii=0; ii<ndim_; ii++)
                                {
                                        result[ii][0] += weight*rho_[ii]*rho_pos[ii]*rho_neg[ii];
                                }
                        }
                }
                else
                {
                        for (int i=0; i<nAngle+1; i++)
                        {
                                for (int j=0; j<nAngle+1; j++)
                                {
                                        if (i == 0 || i == nAngle)
                                        {
                                                if (j == 0 || j == nAngle)
                                                {
                                                        weight = 1./(4*nAngle2);
                                                }
                                                else
                                                {
                                                        weight = 1./(2*nAngle2);
                                                }
                                        }
                                        else if (j == 0 || j == nAngle)
                                        {
                                                weight = 1./(2*nAngle2);
                                        }
                                        else
                                        {
                                                weight = 1./(nAngle2);
                                        }

                                        phi = 2*M_PI/nAngle*i;
                                        theta = M_PI/(2*nAngle)*j;
                                        r[0] = scale*sin(theta)*cos(phi)/l_;
                                        r[1] = scale*sin(theta)*sin(phi)/l_;
                                        r[2] = scale*cos(theta)/l_;

                                        shift(rho_pos, rho_neg, r);

                                        for (int ii=0; ii<ndim_; ii++)
                                        {
                                                result[ii][0] += weight*rho_[ii]*rho_pos[ii]
                                                                 *rho_neg[ii];
                                        }
                                }
                        }
                }

                fftw_execute(p_linecorr);
                double l = pow(scale/l_,3./2*dim_)*result[0][0]/ndim_;

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

        fftw_free(result);
        fftw_free(rho_pos);
        fftw_free(rho_neg);
        fftw_destroy_plan(p_linecorr);
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
                                        ii_pos = util_.VecId((R[0][id_shell[j]]+R[0][ii])%n_, (R[1][id_shell[j]]+R[1][ii])%n_,
                                                            (R[2][id_shell[j]]+R[2][ii])%n_);
                                        ii_neg = util_.VecId((-R[0][id_shell[j]]+R[0][ii]+n_)%n_, (-R[1][id_shell[j]]+R[1][ii]+n_)%n_,
                                                            (-R[2][id_shell[j]]+R[2][ii]+n_)%n_);
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
        double scale = rmin;
        vector<idpair> idmod;
        vector<double> mod;

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "ERROR. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        id_mod(idmod);
        for (int ii=0; ii<fndim_; ii++)
        {
                mod.push_back(idmod[ii].first);
        }

        whiten(1e-7);

        cout << "\t Scale [l_]" << "\t\t Line Correlation" << endl;
        cout << "----------------------------------------------------------" << endl;

        while (scale < rmax)
        {
                int nmax = lower_bound(mod.begin(), mod.end(), 2*M_PI/scale) - mod.begin();
                double l = 0.;

                for (int ii=0; ii<nmax; ii++)
                {
                        for (int jj=0; jj<nmax; jj++)
                        {
                                vector<int> Iii = util_.fCoordId(idmod[ii].second);
                                vector<int> Ijj = util_.fCoordId(idmod[jj].second);
                                int *Iij = new int[dim_];

                                vector<int> kii, kjj;
                                double k2 = mod[ii]*mod[ii];
                                double q2 = mod[jj]*mod[jj];

                                fftw_complex frho_q, frho_kq;
                                double mu = 0.;

                                for (int d=0; d<dim_; d++)
                                {
                                        kii.push_back(util_.i_to_m(Iii[d]));
                                        kjj.push_back(util_.i_to_m(Ijj[d]));
                                        mu += kii[d]*kjj[d];
                                }

                                if (mu < (1.-k2-q2)/2)
                                {
                                        double kqminus = sqrt(k2+q2-2*mu);
                                        for (int d=0; d<dim_; d++)
                                        {
                                                Iij[d] = util_.m_to_i(kii[d]+kjj[d]);
                                        }
                                        frho_kq[0] =  frho_[util_.fVecId(Iij)][0];
                                        frho_kq[1] =  -frho_[util_.fVecId(Iij)][1];
                                        l += sinc(kqminus*scale)*prod3(frho_[idmod[ii].second],
                                                                       frho_[idmod[jj].second], frho_kq);
                                }

                                if (mu > -(1.-k2-q2)/2)
                                {
                                        double kqplus = sqrt(k2+q2+2*mu);
                                        int *Iij = new int[dim_];
                                        if (-kii[dim_-1]+kjj[dim_-1] >= 0)
                                        {
                                                for (int d=0; d<dim_; d++)
                                                {
                                                        Iij[d] = util_.m_to_i(-kii[d]+kjj[d]);
                                                }
                                                frho_q[0] =  frho_[idmod[jj].second][0];
                                                frho_q[1] =  -frho_[idmod[jj].second][1];
                                                l += sinc(kqplus*scale)*prod3(frho_[idmod[ii].second],
                                                                              frho_q, frho_[util_.fVecId(Iij)]);
                                        }
                                        else
                                        {
                                                for (int d=0; d<dim_; d++)
                                                {
                                                        Iij[d] = util_.m_to_i(kii[d]-kjj[d]);
                                                }
                                                frho_q[0] =  frho_[idmod[jj].second][0];
                                                frho_q[1] =  -frho_[idmod[jj].second][1];
                                                frho_kq[0] =  frho_[util_.fVecId(Iij)][0];
                                                frho_kq[1] =  -frho_[util_.fVecId(Iij)][1];
                                                l += sinc(kqplus*scale)*prod3(frho_[idmod[ii].second],
                                                                              frho_q, frho_kq);
                                        }
                                }
                        }
                }

                l *= 2*pow(scale/l_,3./2*dim_);

                out << scale << "\t" << l << endl;
                cout << "\t" << fixed << scale << "\t\t" << fixed << l << endl;

                scale += dr;
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
                frho2_[ii] = (re*re+im*im)/pow(window,4);
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
                        // B *= pow(kf_,4)/(4*M_PI*k3_rel*scale*pow(dk,3));
                        B /= nTriangle_[nr];
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
