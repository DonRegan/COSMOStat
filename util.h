#ifndef UTIL_H
#define UTIL_H

#include <fftw3.h>

class UTIL{

 private:

  int dim_;
  int n_, n2_, ndim_, fn_, fn2_, fndim_;

 public:

  UTIL ();
  UTIL (int, int);
  ~UTIL ();

  void set_grid (int, int);

  int CoordId (int, int);
  int fCoordId (int, int);
  int VecId (int, int);
  int VecId (int, int, int);
  int VecId (int*);
  int distPBC(int);

  int x (int);
  int y (int);
  int z (int);
  int fx (int);
  int fy (int);
  int fz (int);

  int i_to_m (int);
  int m_to_i (int);

  void fftw_normalize (fftw_complex*);
};


double sinc(double);

#endif
