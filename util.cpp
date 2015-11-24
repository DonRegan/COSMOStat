#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "util.h"

using namespace std;


/*======= UTIL =====================================================================*/

UTIL::UTIL () {}

UTIL::UTIL (int numDimension, int numGrid)
{
  DIM = numDimension;
  N = numGrid;
  N2 = N*N;
  NDIM = pow(N,DIM);
  FN = N/2+1;
  FN2 = N*FN;
  FNDIM = pow(N,DIM-1)*FN;
}

UTIL::~UTIL () {}

void UTIL::set_grid (int numDimension, int numGrid)
{
  DIM = numDimension;
  N = numGrid;
  N2 = N*N;
  NDIM = pow(N,DIM);
  FN = N/2+1;
  FN2 = N*FN;
  FNDIM = pow(N,DIM-1)*FN;
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

int UTIL::VecId (int x, int y)
{
  return N*y+x;
}

int UTIL::VecId (int x, int y, int z)
{
  return N2*z+N*y+x;
}

int UTIL::VecId (int *x)
{
  if (DIM == 2)
    {
      return N*x[1]+x[0];
    }
  else
    {
      return N2*x[2]+N*x[1]+x[0];
    }
}

int UTIL::distPBC (int x)
{
  if (x > N/2)
    {
      return N-x;
    }
  else
    {
      return x;
    }
}

int UTIL::x(int index)
{
  if (DIM == 2)
    {
      return index % N;
    }
  else
    {
      return (index % N2) % N;
    }
}

int UTIL::y(int index)
{
  if (DIM == 2)
    {
      return floor(index/N);
    }
  else
    {
      return floor((index % N2)/N);
    }
}

int UTIL::z(int index)
{
  return floor(index/N2);
}

int UTIL::fx(int index)
{
  if (DIM == 2)
    {
      return index % FN;
    }
  else
    {
      return (index % FN2) % FN;
    }
}

int UTIL::fy(int index)
{
  if (DIM == 2)
    {
      return floor(index/FN);
    }
  else
    {
      return floor((index % FN2)/FN);
    }
}

int UTIL::fz(int index)
{
  return floor(index/FN2);
}

int UTIL::i_to_m (int id)
{
  if (id > N/2)
    {
      return id-N;
    }
  else
    {
      return id;
    }
}

int UTIL::m_to_i (int mode)
{
  if (mode >= N/2)
    {
      mode -= N/2;
    }
  else if (mode <= -N/2)
    {
      mode += N/2;
    }

  if (mode >= 0)
    {
      return mode;
    }
  else
    {
      return mode+N;
    }
}


void UTIL::fftw_normalize (fftw_complex *ffield)
{
  for (int ii=0; ii<FNDIM; ii++)
    {
      ffield[ii][0] /= NDIM;
      ffield[ii][1] /= NDIM;
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
