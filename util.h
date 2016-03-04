#ifndef UTIL_H
#define UTIL_H

#include <fftw3.h>
#include <vector>

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
  std::vector<int> fCoordId (int);
  int VecId (int, int);
  int VecId (int, int, int);
  int VecId (int*);
  int fVecId (int*);
  int distPBC (int);

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


double sinc (double);

void prod (fftw_complex, fftw_complex, fftw_complex);
double prod3 (fftw_complex, fftw_complex, fftw_complex);


typedef std::pair<double, int> idpair;
bool comparator (const idpair&, const idpair&);


#define IM1 2147483563
#define IM2 2147483399
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define IMM1 IM1-1
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define AM (1.0/IM1)

float ran2(long*);

#endif
