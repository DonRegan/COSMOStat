#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "util.h"

using namespace std;


/*======= UTIL =====================================================================*/

UTIL::UTIL () {}

UTIL::UTIL (int num_dimension, int num_grid)
{
  dim_ = num_dimension;
  n_ = num_grid;
  n2_ = n_*n_;
  ndim_ = pow(n_,dim_);
  fn_ = n_/2+1;
  fn2_ = n_*fn_;
  fndim_ = pow(n_,dim_-1)*fn_;
}

UTIL::~UTIL () {}

void UTIL::set_grid (int num_dimension, int num_grid)
{
  dim_ = num_dimension;
  n_ = num_grid;
  n2_ = n_*n_;
  ndim_ = pow(n_,dim_);
  fn_ = n_/2+1;
  fn2_ = n_*fn_;
  fndim_ = pow(n_,dim_-1)*fn_;
}

int UTIL::CoordId (int index, int d)
{
  switch (d)
    {
    case 0:
      return x(index);
      break;
    case 1:
      return y(index);
      break;
    case 2:
      return z(index);
      break;
    default:
      cout << "ERROR. Invalid index." << endl;
      exit(EXIT_FAILURE);
    }
}

int UTIL::fCoordId (int index, int d)
{
  switch (d)
    {
    case 0:
      return fx(index);
      break;
    case 1:
      return fy(index);
      break;
    case 2:
      return fz(index);
      break;
    default:
      cout << "ERROR. Invalid index." << endl;
      exit(EXIT_FAILURE);
    }
}

vector<int> UTIL::fCoordId (int index)
{
  vector<int> fcoord;
  for (int d=0; d<dim_; d++)
  {
    fcoord.push_back(fCoordId(index, d));
  }
  return fcoord;
}

int UTIL::VecId (int x, int y)
{
  return n_*y+x;
}

int UTIL::VecId (int x, int y, int z)
{
  return n2_*z+n_*y+x;
}

int UTIL::VecId (int *x)
{
  if (dim_ == 2)
    {
      return n_*x[1]+x[0];
    }
  else
    {
      return n2_*x[2]+n_*x[1]+x[0];
    }
}

int UTIL::fVecId (int x, int y)
{
  return fn_*y+x;
}

int UTIL::fVecId (int x, int y, int z)
{
  return fn2_*z+fn_*y+x;
}

int UTIL::fVecId (int *x)
{
  if (dim_ == 2)
    {
      return fn_*x[1]+x[0];
    }
  else
    {
      return fn2_*x[2]+fn_*x[1]+x[0];
    }
}

int UTIL::fVecId (vector<int> x)
{
  if (dim_ == 2)
    {
      return fn_*x[1]+x[0];
    }
  else
    {
      return fn2_*x[2]+fn_*x[1]+x[0];
    }
}

int UTIL::distPBC (int x)
{
  if (x > n_/2)
    {
      return n_-x;
    }
  else
    {
      return x;
    }
}

int UTIL::x(int index)
{
  if (dim_ == 2)
    {
      return index % n_;
    }
  else
    {
      return (index % n2_) % n_;
    }
}

int UTIL::y(int index)
{
  if (dim_ == 2)
    {
      return floor(index/n_);
    }
  else
    {
      return floor((index % n2_)/n_);
    }
}

int UTIL::z(int index)
{
  return floor(index/n2_);
}

int UTIL::fx(int index)
{
  if (dim_ == 2)
    {
      return index % fn_;
    }
  else
    {
      return (index % fn2_) % fn_;
    }
}

int UTIL::fy(int index)
{
  if (dim_ == 2)
    {
      return floor(index/fn_);
    }
  else
    {
      return floor((index % fn2_)/fn_);
    }
}

int UTIL::fz(int index)
{
  return floor(index/fn2_);
}

int UTIL::i_to_m (int id)
{
  if (id > n_/2)
    {
      return id-n_;
    }
  else
    {
      return id;
    }
}

int UTIL::m_to_i (int mode)
{
  if (mode >= n_/2)
    {
      mode -= n_/2;
    }
  else if (mode <= -n_/2)
    {
      mode += n_/2;
    }

  if (mode >= 0)
    {
      return mode;
    }
  else
    {
      return mode+n_;
    }
}


void UTIL::fftw_normalize (fftw_complex *ffield)
{
  for (int ii=0; ii<fndim_; ii++)
    {
      ffield[ii][0] /= ndim_;
      ffield[ii][1] /= ndim_;
    }
}


double sinc(double x)
{
  if (x == 0)
    {
      return 1.0;
    }
  else
    {
      return sin(x)/x;
    }
}


void cscalarprod (fftw_complex a, double b, fftw_complex result)
{
  result[0] += b*a[0];
  result[1] += b*a[1];
}


void csum (fftw_complex a, fftw_complex b, fftw_complex result)
{
  result[0] = a[0]+b[0];
  result[1] = a[1]+b[1];
}


void prod (fftw_complex a, fftw_complex b, fftw_complex result)
{
  result[0] = a[0]*b[0] - a[1]*b[1];
  result[1] = a[0]*b[1] + a[1]*b[0];
}

double prod3 (fftw_complex a, fftw_complex b, fftw_complex c)
{
  fftw_complex buffer;
  prod(a, b, buffer);
  return buffer[0]*c[0] - buffer[1]*c[1];
}


bool comparator (const idpair &l, const idpair &r)
{
  return l.first < r.first;
}


float ran2(long *idum)
{
  int j;
  long k;
  static long idum2=123456789;
  static long iy=0;
  static long iv[NTAB];
  float temp;
  if (*idum <= 0) {
    if (-(*idum) < 1) *idum=1;
    else *idum = -(*idum);
    idum2=(*idum);
    for (j=NTAB+7;j>=0;j--) {
      k=(*idum)/IQ1;
      *idum=IA1*(*idum-k*IQ1)-k*IR1;
      if (*idum < 0) *idum += IM1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ1;
  *idum=IA1*(*idum-k*IQ1)-k*IR1;
  if (*idum < 0) *idum += IM1;
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2;
  if (idum2 < 0) idum2 += IM2;
  j=iy/NDIV;
  iy=iv[j]-idum2;
  iv[j] = *idum;
  if (iy < 1) iy += IMM1;
  if ((temp=AM*iy) > RNMX) return RNMX;
  else return temp;
}
