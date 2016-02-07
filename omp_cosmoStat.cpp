#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <complex>
#include <vector>
#include <stdlib.h>
//#include <omp.h>
#include "util.h"
#include "omp_cosmoStat.h"

using namespace std;


/* +++++++++++++++++++++++++++++++++ CONSTRUCTORS +++++++++++++++++++++++++++++++++++*/

COSMOStat::COSMOStat () {
}


COSMOStat::COSMOStat (int numDimension, int numGrid, double boxsize)
{
        if (numDimension != 2 && numDimension != 3)
        {
                cout << "ERROR. The supported number of dimensions are either 2 or 3." << endl;
                exit(EXIT_FAILURE);
        }

        DIM = numDimension;
        N = numGrid;
        NDIM = pow(numGrid,numDimension);
        FNDIM = pow(numGrid,numDimension-1)*(int(N/2)+1);
        L = boxsize;
        kF = 2*M_PI/L;

        util.set_grid(DIM, N);

        rho = new double[NDIM];
        frho = new fftw_complex[FNDIM];
        frho2 = new double[FNDIM];
        K = new vector<int>[DIM];

        if (DIM == 2)
        {
#pragma omp critical (make_plan)
                {
                        p_rho = fftw_plan_dft_r2c_2d(N, N, rho, frho, FFTW_MEASURE);
                        ip_rho = fftw_plan_dft_c2r_2d(N, N, frho, rho, FFTW_MEASURE);
                }
        }
        else
        {
#pragma omp critical (make_plan)
                {
                        p_rho = fftw_plan_dft_r2c_3d(N, N, N, rho, frho, FFTW_MEASURE);
                        ip_rho = fftw_plan_dft_c2r_3d(N, N, N, frho, rho, FFTW_MEASURE);
                }
        }

        for (int ii=0; ii<FNDIM; ii++)
        {
                double k2 = 0.;
                for (int d=0; d<DIM; d++)
                {
                        K[d].push_back(util.i_to_m(util.fCoordId(ii,d)));
                        k2 += pow(K[d][ii],2);
                }
                AbsK.push_back(sqrt(k2)*kF);
        }

        for (int ii=0; ii<NDIM; ii++)
        {
                rho[ii] = 0.;
        }
}


COSMOStat::~COSMOStat ()
{
        delete [] rho;
        delete [] frho;
        delete [] frho2;

        fftw_destroy_plan(p_rho);
        fftw_destroy_plan(ip_rho);
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
                if (ii == NDIM)
                {
                        cout << "ERROR. Wrong file format." << endl;
                        exit(EXIT_FAILURE);
                }

                rho[ii] = buffer;
                ii++;
        }

        data.close();

        fftw_execute(p_rho);
        util.fftw_normalize(frho);
}


void COSMOStat::cic (particle_data_pos *P, int NumPart)
{
        double spacing = L/N;
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

                int ipp = (i+1)%N;
                int jpp = (j+1)%N;
                int kpp = (k+1)%N;

                rho[util.VecId(i,j,k)] += tx*ty*tz;
                rho[util.VecId(ipp,j,k)] += dx*ty*tz;
                rho[util.VecId(i,jpp,k)] += tx*dy*tz;
                rho[util.VecId(i,j,kpp)] += tx*ty*dz;
                rho[util.VecId(ipp,jpp,k)] += dx*dy*tz;
                rho[util.VecId(ipp,j,kpp)] += dx*ty*dz;
                rho[util.VecId(i,jpp,kpp)] += tx*dy*dz;
                rho[util.VecId(ipp,jpp,kpp)] += dx*dy*dz;
        }
}


void COSMOStat::rho2delta ()
{
        double rhoAvg = 0.;
        for (int ii=0; ii<NDIM; ii++)
        {
                rhoAvg += rho[ii];
        }
        rhoAvg /= NDIM;
        for (int ii=0; ii<NDIM; ii++)
        {
                rho[ii] = rho[ii]/rhoAvg-1;
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

        if (DIM == 2)
        {
                for (int i=0; i<N; i++)
                {
                        for (int j=0; j<N; j++)
                        {
                                out << rho[util.VecId(i,j)] << "\t";
                        }
                        out << endl;
                }
        }
        else
        {
                for (int ii=0; ii<NDIM; ii++)
                {
                        out << rho[ii] << endl;
                }
        }

        out.close();
}


void COSMOStat::save_slice(string fname, int d, int val)
{
        int ii;
        fstream out;
        out.open(fname.c_str(), ios::out);

        for (int i=0; i<N; i++)
        {
                for (int j=0; j<N; j++)
                {
                        switch (d)
                        {
                        case 1:
                                ii = util.VecId(val, i, j);
                                break;
                        case 2:
                                ii = util.VecId(i, val, j);
                                break;
                        case 3:
                                ii = util.VecId(i, j, val);
                                break;
                        default:
                                cout << "Error. Invalid dimension - "
                                     << "choose either 1 (x), 2 (y) or 3 (z)." << endl;
                                exit(EXIT_FAILURE);
                        }
                        out << rho[ii] << "\t";
                }
                out << endl;
        }

        out.close();
}


void COSMOStat::set_Rho (double val, int id)
{
        rho[id] = val;
}



void COSMOStat::set_RhoSubCube (double *parentRho, int parentN, int subId)
{
        int x[DIM], subx[DIM];
        double kf = 2*M_PI/L;
        UTIL parentGrid(DIM, parentN), subGrid(DIM, parentN/N);

        for (int d=0; d<DIM; d++)
        {
                subx[d] = subGrid.CoordId(subId,d)*N;
        }
        for (int ii=0; ii<NDIM; ii++)
        {
                for (int d=0; d<DIM; d++)
                {
                        x[d] = util.CoordId(ii,d)+subx[d];
                }
                rho[ii] = parentRho[parentGrid.VecId(x)];
        }

        do_FFT();

        for (int ii=0; ii<FNDIM; ii++)
        {
                double window = 1.;
                for (int d=0; d<DIM; d++)
                {
                        window *= sinc(M_PI*K[d][ii]/N);
                }
                double re = frho[ii][0];
                double im = frho[ii][1];
                frho2[ii] = (re*re+im*im)/pow(window,4);
        }
}


/* +++++++++++++++++++++++++++++++++ SUBROUTINES ++++++++++++++++++++++++++++++++++++*/

void COSMOStat::shift (fftw_complex *field, double *dx)
{
        fftw_complex *ffield = new fftw_complex[NDIM];
        fftw_plan p, ip;

        if (DIM == 2)
        {
                ip = fftw_plan_dft_2d(N, N, ffield, field, FFTW_BACKWARD, FFTW_ESTIMATE);
        }
        else
        {
                ip = fftw_plan_dft_3d(N, N, N, ffield, field, FFTW_BACKWARD, FFTW_ESTIMATE);
        }

        for (int ii=0; ii<NDIM; ii++)
        {
                double phase = 0.;
                for (int d=0; d<DIM; d++)
                {
                        phase += K[d][ii]*dx[d];
                }
                phase *= 2*M_PI;
                double CosPhase = cos(phase);
                double SinPhase = sin(phase);
                double re = frho[ii][0]*CosPhase-frho[ii][1]*SinPhase;
                double im = frho[ii][0]*SinPhase+frho[ii][1]*CosPhase;
                ffield[ii][0] = re;
                ffield[ii][1] = im;
        }

        fftw_execute(ip);

        fftw_free(ffield);
        fftw_destroy_plan(ip);
}


void COSMOStat::shift (double *field_pos, double *field_neg, double *dx)
{
        fftw_complex *ffield_pos = new fftw_complex[FNDIM];
        fftw_complex *ffield_neg = new fftw_complex[FNDIM];
        fftw_plan ip_pos, ip_neg;

        if (DIM == 2)
        {
#pragma omp critical (make_plan)
                {
                        ip_pos = fftw_plan_dft_c2r_2d(N, N, ffield_pos, field_pos, FFTW_ESTIMATE);
                        ip_neg = fftw_plan_dft_c2r_2d(N, N, ffield_neg, field_neg, FFTW_ESTIMATE);
                }
        }
        else
        {
#pragma omp critical (make_plan)
                {
                        ip_pos = fftw_plan_dft_c2r_3d(N, N, N, ffield_pos, field_pos, FFTW_ESTIMATE);
                        ip_neg = fftw_plan_dft_c2r_3d(N, N, N, ffield_neg, field_neg, FFTW_ESTIMATE);
                }
        }

        for (int ii=0; ii<FNDIM; ii++)
        {
                double phase = 0.;
                for (int d=0; d<DIM; d++)
                {
                        phase += K[d][ii]*dx[d];
                }
                phase *= 2*M_PI;
                double CosPhase = cos(phase);
                double SinPhase = sin(phase);
                double re1 = frho[ii][0]*CosPhase;
                double re2 = frho[ii][1]*SinPhase;
                double im1 = frho[ii][0]*SinPhase;
                double im2 = frho[ii][1]*CosPhase;
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
        for (int ii=0; ii<FNDIM; ii++)
        {
                double re = frho[ii][0], im = frho[ii][1];
                double mag = sqrt(re*re+im*im), k2 = 0.;
                for (int d=0; d<DIM; d++)
                {
                        k2 += pow(K[d][ii],2);
                }
                if (ii == 0 || mag < threshold)
                {
                        frho[ii][0] = 0.;
                        frho[ii][1] = 0.;
                }
                else
                {
                        frho[ii][0] *= exp(-pow(0*2*M_PI/N,2)/2*k2)/mag;
                        frho[ii][1] *= exp(-pow(0*2*M_PI/N,2)/2*k2)/mag;
                }
        }
        fftw_execute(ip_rho);
        fftw_execute(p_rho);
        util.fftw_normalize(frho);
}


void COSMOStat::filter (double scale, short filterMode)
{
        switch (filterMode)
        {
        case 1: // Top hat spherical cutoff
                for (int ii=0; ii<FNDIM; ii++)
                {
                        double k2 = 0.;
                        for (int d=0; d<DIM; d++)
                        {
                                k2 += pow(K[d][ii],2);
                        }
                        double cutoff_scale = pow(scale/L,2)*k2;
                        if (cutoff_scale > 1.)
                        {
                                frho[ii][0] = 0.;
                                frho[ii][1] = 0.;
                        }
                }
                break;
        default:
                cout << "Error. Cutoff mode not defined." << endl;
                exit(EXIT_FAILURE);
        }
        fftw_execute(ip_rho);
        fftw_execute(p_rho);
        util.fftw_normalize(frho);
}


void COSMOStat::shell_c2r (double *rho_shell, double scale, double binSize)
{
        fftw_complex *frho_shell = new fftw_complex[FNDIM];
        fftw_plan ip_rho_shell;

        if (DIM == 2)
        {
#pragma omp critical (make_plan)
                {
                        ip_rho_shell = fftw_plan_dft_c2r_2d(N, N, frho_shell, rho_shell,
                                                            FFTW_ESTIMATE);
                }
        }
        else
        {
#pragma omp critical (make_plan)
                {
                        ip_rho_shell = fftw_plan_dft_c2r_3d(N, N, N, frho_shell, rho_shell,
                                                            FFTW_ESTIMATE);
                }
        }

        for (int ii=0; ii<FNDIM; ii++)
        {
                if (AbsK[ii] > scale-binSize/2 && AbsK[ii] < scale+binSize/2)
                {
                        frho_shell[ii][0] = frho[ii][0];
                        frho_shell[ii][1] = frho[ii][1];
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
                shell[i] = new double[NDIM];
                fshell[i] = new fftw_complex[FNDIM];
        }

        if (DIM == 2)
        {
                for (int i=0; i<3; i++)
                {
                        ip_shell[i] = fftw_plan_dft_c2r_2d(N, N, fshell[i], shell[i],
                                                           FFTW_ESTIMATE);
                }
        }
        else
        {
                for (int i=0; i<3; i++)
                {
                        ip_shell[i] = fftw_plan_dft_c2r_3d(N, N, N, fshell[i], shell[i],
                                                           FFTW_ESTIMATE);
                }
        }

        while (scale < kmax)
        {
                k[0] = scale;
                k[1] = k2_rel*scale;
                k[2] = k3_rel*scale;

                for (int ii=0; ii<FNDIM; ii++)
                {
                        for (int i=0; i<3; i++)
                        {
                                if (AbsK[ii] > k[i]-dk/2 && AbsK[ii] < k[i]+dk/2)
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
                for (int ii=0; ii<NDIM; ii++)
                {
                        ntr += shell[0][ii]*shell[1][ii]*shell[2][ii];
                }

                nTriangle.push_back(int(ntr/NDIM));
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


double COSMOStat::get_RhoAvg ()
{
        double rhoAvg = 0.;
        for (int ii=0; ii<NDIM; ii++)
        {
                rhoAvg += rho[ii];
        }
        return rhoAvg/NDIM;
}


double COSMOStat::get_PowerSpec (double k, double dk)
{
        double power = 0.;
        int nk = 0;
        for (int ii=0; ii<FNDIM; ii++)
        {
                if (AbsK[ii] > k-dk/2 && AbsK[ii] < k+dk/2)
                {
                        power += frho2[ii];
                        nk++;
                }
        }
        power *= pow(L,DIM)/nk;

        return power;
}


/* +++++++++++++++++++++++++++++++++ ESTIMATORS +++++++++++++++++++++++++++++++++++++*/

void COSMOStat::compute_LineCorr (string fname, double rmin, double rmax, double dr, short filterMode)
{
        fftw_complex *result = new fftw_complex[NDIM];
        double *rho_pos = new double[NDIM];
        double *rho_neg = new double[NDIM];
        fftw_plan p_linecorr;

        double *r = new double[DIM];
        double scale = rmin, phi, theta, weight;
        int nAngle, nAngle2;

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "ERROR. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        if (DIM == 2)
        {
#pragma omp critical (make_plan)
                {
                        p_linecorr = fftw_plan_dft_2d(N, N, result, result,
                                                      FFTW_FORWARD, FFTW_MEASURE);
                }
        }
        else
        {
#pragma omp critical (make_plan)
                {
                        p_linecorr = fftw_plan_dft_3d(N, N, N, result, result,
                                                      FFTW_FORWARD, FFTW_MEASURE);
                }
        }

        whiten(1e-7);

        cout << "\t Scale [L]" << "\t\t Line Correlation" << endl;
        cout << "----------------------------------------------------------" << endl;

        while (scale < rmax)
        {
                nAngle = (floor(scale*N/L/10.)+1)*20;
                nAngle2 = nAngle*nAngle;
                for (int ii=0; ii<NDIM; ii++)
                {
                        result[ii][0] = 0.;
                        result[ii][1] = 0.;
                }

                filter(scale, filterMode);

                if (DIM == 2)
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
                                r[0] = scale*cos(phi)/L;
                                r[1] = scale*sin(phi)/L;

                                shift(rho_pos, rho_neg, r);

                                for (int ii=0; ii<NDIM; ii++)
                                {
                                        result[ii][0] += weight*rho[ii]*rho_pos[ii]*rho_neg[ii];
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
                                        r[0] = scale*sin(theta)*cos(phi)/L;
                                        r[1] = scale*sin(theta)*sin(phi)/L;
                                        r[2] = scale*cos(theta)/L;

                                        shift(rho_pos, rho_neg, r);

                                        for (int ii=0; ii<NDIM; ii++)
                                        {
                                                result[ii][0] += weight*rho[ii]*rho_pos[ii]
                                                                 *rho_neg[ii];
                                        }
                                }
                        }
                }

                fftw_execute(p_linecorr);
                double l = pow(scale/L,3./2*DIM)*result[0][0]/NDIM;

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
        double scale = rmin, rf = L/N;
        int Nr, binSize = 2;

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "ERROR. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        vector<double> absR;
        vector<int> *R = new vector<int>[DIM];
        for (int ii=0; ii<NDIM; ii++)
        {
                double r2 = 0.;
                for (int d=0; d<DIM; d++)
                {
                        R[d].push_back(util.CoordId(ii,d));
                        r2 += pow(util.CoordId(ii,d),2);
                }
                absR.push_back(sqrt(r2)*rf);
        }

        whiten(1e-7);

        cout << "\t Scale [L]" << "\t\t Line Correlation" << endl;
        cout << "----------------------------------------------------------" << endl;

        while (scale < rmax)
        {
                int *id_dist = new int[DIM];
                vector<int> id_shell;
                double id_scale = scale/rf, id_scale_p = scale/rf+0.5*binSize, id_scale_m = sqrt(2)/2*(scale/rf-0.5*binSize);
                double l = 0.;

                filter(scale, filterMode);

                Nr = 0;
                for (int ii=0; ii<FNDIM; ii++)
                {
                        int b1 = 1, b2 = 1;
                        for (int d=0; d<DIM; d++)
                        {
                                id_dist[d] = util.distPBC(R[d][ii]);
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
                                for (int d=0; d<DIM; d++)
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

                for (int ii=0; ii<NDIM; ii++)
                {
                        double avg = 0.;
                        for (int j=0; j<Nr; j++)
                        {
                                int ii_pos, ii_neg;
                                if (DIM == 2)
                                {
                                        ii_pos = util.VecId((R[0][id_shell[j]]+R[0][ii])%N, (R[1][id_shell[j]]+R[1][ii])%N);
                                        ii_neg = util.VecId(((-R[0][id_shell[j]]+R[0][ii]+N)%N), ((-R[1][id_shell[j]]+R[1][ii]+N)%N));
                                }
                                else
                                {
                                        ii_pos = util.VecId((R[0][id_shell[j]]+R[0][ii])%N, (R[1][id_shell[j]]+R[1][ii])%N,
                                                            (R[2][id_shell[j]]+R[2][ii])%N);
                                        ii_neg = util.VecId((-R[0][id_shell[j]]+R[0][ii]+N)%N, (-R[1][id_shell[j]]+R[1][ii]+N)%N,
                                                            (-R[2][id_shell[j]]+R[2][ii]+N)%N);
                                }
                                avg += rho[ii_pos]*rho[ii_neg];
                        }

                        l += rho[ii]*avg;
                }

                l *= pow(scale/L,3./2*DIM)/NDIM/Nr;

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

        for (int ii=0; ii<FNDIM; ii++)
        {
                double window = 1.;
                for (int d=0; d<DIM; d++)
                {
                        window *= sinc(M_PI*K[d][ii]/N);
                }
                double re = frho[ii][0];
                double im = frho[ii][1];
                frho2[ii] = (re*re+im*im)/pow(window,4);
        }

        cout << "\t Scale [1/L]" << "\t\t Power Spectrum" << endl;
        cout << "----------------------------------------------------------" << endl;

        while (scale < kmax)
        {
                double power = 0.;
                int nk = 0;
                for (int ii=0; ii<FNDIM; ii++)
                {
                        if (AbsK[ii] > scale-dk/2 && AbsK[ii] < scale+dk/2)
                        {
                                power += frho2[ii];
                                nk++;
                        }
                }
                power *= pow(L,DIM)/nk;

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
        double *rho_shell = new double[NDIM];

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "ERROR. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        if (kmin < kF)
        {
                cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/L."
                     << endl;
                kmin = kF;
                scale = kmin;
        }
        if (kmax > N*kF/2)
        {
                cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = N*Pi/L."
                     << endl;
                kmax = N*kF/2;
        }

        cout << "\t k [1/L]" << "\t\t Power Spectrum [L^DIM]" << endl;
        cout << "----------------------------------------------------------" << endl;

        while (scale < kmax)
        {
                shell_c2r(rho_shell, scale, dk);

                double power = 0.;
                for (int ii=0; ii<NDIM; ii++)
                {
                        power += pow(rho_shell[ii],2);
                }

                if (DIM == 2)
                {
                        power *= pow(kF,2)/(2*M_PI*scale*(2*kF));
                }
                else
                {
                        power *= pow(kF,3)/(4*M_PI*pow(scale,2)*2*kF);
                }

                power *= pow(L,DIM)/NDIM;

                out << scale << "\t" << power << endl;
                cout << "\t " << fixed << scale << "\t\t " << fixed << power << endl;

                scale += dk;
        }

        cout << "----------------------------------------------------------" << endl;
}


void COSMOStat::compute_PositionDependentPowerSpec (string fname, double kmin, double kmax, double dk,
                                                    int nCut, int subId)
{
        COSMOStat subBox(DIM, N/nCut, L/nCut);
        subBox.set_RhoSubCube(rho, N, subId);
        subBox.do_FFT();
        subBox.compute_PowerSpec(fname, kmin, kmax, dk);
}


void COSMOStat::compute_IntegratedBiSpec (string fname, double kmin, double kmax, double dk, int nCut)
{
        double scale = kmin, kf = 2*M_PI/(L/nCut);
        COSMOStat subBox(DIM, N/nCut, L/nCut);

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "ERROR. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        if (kmin < kf)
        {
                cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/(L/nCut)."
                     << endl;
                kmin = kf;
                scale = kmin;
        }
        if (kmax > N*kf/2)
        {
                cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = N*Pi/L."
                     << endl;
                kmax = N*kf/2;
        }

        cout << "\t k [1/L]" << "\t\t Integrated Bispectrum [V^2]" << endl;
        cout << "----------------------------------------------------------" << endl;

        vector<double> iB;
        while (scale < kmax)
        {
                iB.push_back(0.0);
                scale += dk;
        }
        scale = kmin;

        for (int i=0; i<pow(nCut,DIM); i++)
        {
                subBox.set_RhoSubCube(rho, N, i);
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
                iB[i] /= pow(nCut,DIM);
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
        double *rho_shell1 = new double[NDIM];
        double *rho_shell2 = new double[NDIM];
        double *rho_shell3 = new double[NDIM];

        fstream out;
        out.open(fname.c_str(), ios::out);

        if (!out)
        {
                cout << "Error. Cannot open file." << endl;
                exit(EXIT_FAILURE);
        }

        if (kmin < kF)
        {
                cout << "WARNING. Value for kmin smaller than fundamental mode. Setting kmin = 2*Pi/L."
                     << endl;
                kmin = kF;
                scale = kmin;
        }
        if (kmax > N*kF/2)
        {
                cout << "WARNING. Value for kmax bigger than the Nyquist frequency. Setting kmax = N*Pi/L."
                     << endl;
                kmax = N*kF/2;
        }

        if (nTriangle_.size() == 0)
        {
          COSMOStat TriMesh(DIM,(1+int(3.*int(dk/kF)*int(kmax/kF)/10))*10, L);
          nTriangle_ = TriMesh.get_nTriangle(kmin, kmax, dk, k2_rel, k3_rel);
        }

        cout << "\t k_1 [1/L]" << "\t\t Bispectrum (k_2 = " << k2_rel << "*k_1, k_3 = "
             << k3_rel << "*k_1)" << endl;
        cout << "----------------------------------------------------------" << endl;

        int nr = 0;
        while (scale < kmax)
        {
                shell_c2r(rho_shell1, scale, dk);
                shell_c2r(rho_shell2, k2_rel*scale, dk);
                shell_c2r(rho_shell3, k3_rel*scale, dk);

                double B = 0.;
                for (int ii=0; ii<NDIM; ii++)
                {
                        B += rho_shell1[ii]*rho_shell2[ii]*rho_shell3[ii];
                }

                if (DIM == 2)
                {
                        // B *= pow(kF,4)/(4*M_PI*k3_rel*scale*pow(dk,3));
                        B /= nTriangle_[nr];
                }
                else
                {
                        // B *= pow(kF,6)/(8*M_PI*M_PI*k2_rel*k3_rel*pow(scale*dk,3));
                        B /= nTriangle_[nr];
                }

                B *= pow(L,2*DIM)/NDIM;

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
//   double *frho2 = new double[NDIM];
//   double scale = kmin, kf = 2*M_PI/L;
//   vector<double> k2;

//   fstream out;
//   out.open(fname.c_str(), ios::out);

//   if (!out)
//     {
//       cout << "Error. Cannot open file." << endl;
//       exit(EXIT_FAILURE);
//     }

//   for (int ii=0; ii<NDIM; ii++)
//     {
//       double re = frho[ii][0];
//       double im = frho[ii][1];
//       frho2[ii] = sqrt(re*re+im*im);
//     }

//   for (int ii=0; ii<NDIM; ii++)
//     {
//       k2.push_back(0.0);
//       for (int d=0; d<DIM; d++)
//  {
//    k2[ii] += pow(K[d][ii],2);
//  }
//       k2[ii] *= pow(kf,2);
//     }

//   cout << "\t Scale [1/L]" << "\t\t Power Spectrum" << endl;
//   cout << "----------------------------------------------------------" << endl;

//   while (scale < kmax)
//     {
//       double power = 0.;
//       int nk = 0;
//       for (int ii=0; ii<NDIM; ii++)
//  {
//    if (sqrt(k2[ii]) > scale-kf/2 && sqrt(k2[ii]) < scale+kf/2)
//      {
//        power += frho2[ii];
//        nk++;
//      }
//  }
//       power /= nk;

//       out << scale << "\t" << power << endl;
//       cout << "\t " << fixed << scale << "\t\t " << fixed << power << endl;

//       scale += 2*M_PI/L;
//     }

//   cout << "----------------------------------------------------------" << endl;
// }
