#ifndef COSMOSTAT_H
#define COSMOSTAT_H

#include <string>
#include <vector>
#include <fftw3.h>
#include "util.h"
#include "proto.h"


class COSMOStat{

 private:

  int dim_;                       //! dimension of the density field
  int n_, ndim_, fndim_;          //! number of grid cells per side, number of total
                                  //!  grid cells in real and Fourier space
  double l_;                      //! physical sidelength of the box
  double kf_;                     //! fundamental mode of the Fourier grid

  UTIL util_;                     //! utility class that handels the gridding in real
                                  //!  and Fourier space

  double *rho_;                   //! array that stores the density field
  fftw_complex *frho_;            //! Fourier transform of the density field
  double *frho2_;                 //! squared amplitudes of the Fourier coefficients

  fftw_plan p_rho_, ip_rho_;      //! forward and backward plans for the FFTW library

  std::vector<int> *idk_;         //! dim-dimensional vector that stores the k-modes in
                                  //!  fundamental units, i.e. kf = 2Pi/L
  std::vector<double> absk_;      //! vector that stores the length of those modes
  std::vector<int> nTriangle_;    //! vector used to estimate the number of triangles in
                                  //!  the bispectrum computation

  long seed_;

  void shift (fftw_complex*, double*);
  void shift (double*, double*, double*);
  void whiten (double);
  void filter (double, short);
  void shell_c2r (double*, double, double);

  /**
   * Function: Get_nTriangle
   * Computes the number of triangle configurations for a given shape k1*(1,k2_rel,k3_rel)
   *  and a range of scales for k1.
   * Parameters:
   *  kmin - minimal value for k1 (in physical units)
   *  kmax - maximal value for k1
   *  dk - linear scale increment and width of the k-bins
   *  k2_rel - ratio of k2/k1
   *  k3_rel - ratio of k3/k1
   * Returns:
   *  int-vector containing the number of triangles for the full range of scales.
   */
  std::vector<int> get_nTriangle (double, double, double, double, double);
  void id_mod (std::vector<idpair>*);


 public:

  COSMOStat ();
  COSMOStat (int, int, double);
  COSMOStat (int, int, double, long);
  ~COSMOStat ();

  /**
   * Function: Load
   * Loads two- or three-dimensional field into member variable rho and computes its
   *  Fourier transform. The data must be in matrix format (2d) or single column format
   *  (3d).
   * Parameters:
   *  fname - string containing the filename to the data
   */
  void load (std::string);

  void load_particles (std::string, int, double);

  /**
   * Function: Cic
   * Takes the (3d) positions of a collection of particles and smoothes them onto a grid
   *  using a cloud-in-cell algorithm. The resulting field is stored in the member
   *  variable rho.
   * Parameters:
   *  P - pointer to an array of particle positions (data format defined in proto.h)
   *  NumPart - number of elements contained in that array
   */
  void cic (particle_data_pos*, int);

  struct particle_data_pos* subsample (particle_data_pos*, int, double, long);

  /**
   * Function: Rho2delta
   * Computes the density constrast from the density field, i.e. delta = rho/rho_avg-1.
   *  In doing so, the member variable rho is overwritten.
   * Parameters:
   *  None.
   */
  void rho2delta ();

  /**
   * Function: Save
   * Saves the field stored in the member variable rho. If the field is 2d, it is stored
   *  in matrix format; if it is 3d, it is stored in a single column format.
   * Parameters:
   *  fname - string containing the filename the data will be saved in.
   */
  void save (std::string);

  /**
   * Function: Save
   * Saves a slice through the field stored in the member variable rho.
   * Parameters:
   *  fname - string containing the filename to the data
   *  d - dimension perpendicular to the slice
   *  val - position of the slice in the perpendicular dimension
   */
  void save_slice (std::string, int, int);

  /**
   * Function: Do_FFT
   * Executes the Fourier transform of the member variable rho and storing it in frho.
   * Parameters:
   *  None.
   */
  void do_FFT()
  {
    fftw_execute(p_rho_);
    util_.fftw_normalize(frho_);
  }

  /**
   * Function: Set_Rho
   * Sets a given position of the member variable rho to a given value.
   * Parameters:
   *  val - value the field will be set to
   *  id - flattened index of the position in the rho-array
   */
  void set_Rho (double, int);

  /**
   * Function: Set_RhoSubCube
   * Takes a cubic subsection of a bigger field and stores it in the member variable
   *  rho. It computes the Fourier transform and deconvolves a cloud-in-cell window
   *  function.
   * Parameters:
   *  parentRho - pointer to the array containing the bigger field
   *  parentN - number of grid cells per side of the bigger field
   *  subId - identifier (in usual flattened index format) of the subcube, there will be
   *   a total number of (parentN/N)^DIM subcubes
   */
  void set_RhoSubCube (double*, int, int);

  /**
   * Function: Get_Rho
   * Returns the value of the member variable rho at a given position.
   * Parameters:
   *  ii - flattened index of the position
   * Returns:
   *  double value of the array
   */
  double get_Rho (int ii) { return rho_[ii]; }

  /**
   * Function: Get_RhoAvg
   * Computes and returns the average of the member variable rho.
   * Parameters:
   *  None.
   * Returns:
   *  double value containing the average
   */
  double get_RhoAvg ();

  /**
   * Function: Get_PowerSpec
   * Computes and returns the power spectrum of the field stored in the member variable
   *  rho at a given scale.
   * Parameters:
   *  k - scale in physical units (h/Mpc)
   *  dk - width of the k-bin in the same units
   * Returns:
   *  double value containing the power
   */
  double get_PowerSpec (double, double);


  void generate_NGfield (int);

  /**
   * Function: Compute_LineCorr
   * Computes and saves the line correlation function within a given range of scales.
   * Parameters:
   *  fname - filename the line correlation function will be saved to
   *  rmin - smallest scale the line correlation function is computed for (in physical
   *   units, i.e. Mpc/h)
   *  rmax - largest scale the line correlation function is computed for
   *  dr - increment for the scale
   *  filterMode - specifies the filter used by the line correlation function, should
   *   be set to '1' for now, which corresponds to a spherical top hat
   */
  void compute_LineCorr (std::string, double, double, double, short);

  /**
   * Function: Compute_LineCorr_2
   * Computes and save the line correlation function within a given range of scales.
   *  This is a different implementation and should not be used for now.
   * Parameters:
   */
  void compute_LineCorr_2 (std::string, double, double, double, short);


  void compute_LineCorr_F (std::string, double, double, double);

  /**
   * Function: Compute_PowerSpec
   * Computes and saves the power spectrum within a given range of scales. The window
   *  function of the cloud-in-cell algorithm is deconvolved beforehand.
   * Parameters:
   *  fname - filename the power spectrum will be saved to
   *  kmin - smallest scale (in physical units)
   *  kmax - largest scale
   *  dk - linear scale increment and width of the k-bins
   */
  void compute_PowerSpec (std::string, double, double, double);

  /**
   * Function: Compute_PowerSpec2
   * Computes and saves the power spectrum within a given range of scales. This is a
   *  different implementation and should not be used for now.
   * Parameters:
   *  fname - filename the power spectrum will be saved to
   *  kmin - smallest scale (in physical units)
   *  kmax - largest scale
   *  dk - linear scale increment and width of the k-bins
   */
  void compute_PowerSpec_2 (std::string, double, double, double);

  /**
   * Function: Compute_PositionDependentPowerSpec
   * Computes and saves the power spectrum within a given range of scales for a subcube
   *  of the density field.
   * Parameters:
   *  fname - filename the power spectrum will be saved to
   *  kmin - smallest scale (in physical units)
   *  kmax - largest scale
   *  dk - linear scale increment and width of the k-bins
   *  nCut - number of subcubes per side
   *  subId - identifier of the subcube for which the power spectrum will be computed
   */
  void compute_PositionDependentPowerSpec (std::string, double, double, double, int, int);

  /**
   * Function: Compute_IntegratedBiSpec
   * Computes and saves the integrated bispectrum within a given range of scales.
   * Parameters:
   *  fname - filename the power spectrum will be saved to
   *  kmin - smallest scale (in physical units)
   *  kmax - largest scale
   *  dk - linear scale increment and width of the k-bins
   *  nCut - number of subcubes per side
   */
  void compute_IntegratedBiSpec (std::string, double, double, double, int);
  void compute_BiSpec (std::string, double, double, double, double, double);
};

#endif
