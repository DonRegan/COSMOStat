#include "omp_cosmoStat.h"
#include "util.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <time.h>
#include <omp.h>
#include <mpi.h>

using namespace std;




/**************************** MAIN ***************************************************/

int main(int argc, char *argv[])
{
  // Location of ZBOX2 data 
  string fin = "/lustre/scratch/astro/res33/ZBOX2/";

  // Output filename
  string fout = "./out";

  // Number of realizations  
  int nF = 1;

  // Number of files per snapshot
  int files = 10;


  // Initialize MPI 
  int nproc, rank, nthread, npxt;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  nthread = 1; // omp_get_num_threads();
  npxt = nproc*nthread;
  
  // Distribute realisations on various processors
  int k, kstart, kend;
  kstart = (nF/npxt)*(rank*nthread);
  if (nF % npxt > rank*nthread)
    {
      kstart += rank*nthread;
      kend = kstart + (nF/npxt)+1;
    }
  else
    {
      kstart += nF % npxt;
      kend = kstart + (nF/npxt);
    }

  // Specify realisations to skip (e.g. damaged files etc.)
  int skip[] = {197,108,146};



  // Loop over realisations on each processor
#pragma omp parallel for
  for (k=kstart; k<kend; k++)
    {
      bool calc = true;
      for (int i=0; i<sizeof(skip)/sizeof(int); i++)
	{
	  if (k+1 == skip[i])
	    {
	      calc = false;
	      break;
	    }
	}

      if (calc){

	// ==================================== LOAD PARTICLE DATA =====================================
	char path[200], input_fname[200], basename[200];
	int snapshot_number, NumPart;

	if (k < 8)
	  {
	    snapshot_number = 16;
	  }
	else if (k < 30)
	  {
	    snapshot_number = 7;
	  }
	else if (k < 40)
	  {
	    snapshot_number = 2;
	  }
	else
	  {
	    snapshot_number = 3;
	  }

	cout << "Rank " << rank << " processes snapshot " << k+1 << endl;

	// Initialize COSMOStat-class, which stores the data and handles all statistics
	COSMOStat stat(3, 1024, 1500.0);

	// Get input filenames
	stringstream fend, snap;
	fend << "LCDM-750-run" << k+1;
	snap << "snap" << snapshot_number;
	sprintf(path, (fin+fend.str()+"/DATA").c_str());
	sprintf(basename, "LCDM-L1500-N750-Tf_om_m_0.25_om_de_0.75_om_b_0.04_sig8_0.8_h_0.7");
	sprintf(input_fname, "%s/%s_%03d", path, basename, snapshot_number);

	// Load particle data into "P" and convert into smooth density field using CIC 
	for (int i=0; i<files; i++)
	  {
	    particle_data_pos *P = load_sub_snapshot(input_fname, i, NumPart);	  
	    stat.cic(P, NumPart);
	    delete [] P;
	  }
	stat.rho2delta();
	stat.do_FFT();
	// =============================================================================================


	// ====================================  MEASURE STATISTICS ====================================

	// Line correlation function
	// stat.compute_LineCorr(fout+fend.str()+".dat", 15., 201., 5., 1);

	// Power Spectrum
	// stat.compute_PowerSpec(fout+fend.str()+".dat", 2*M_PI/1500., 
	// 		          256*M_PI/1500.+2*M_PI/1500., 254*M_PI/1500./22);

	// Bispectrum
	// stat.compute_BiSpec(fout+fend.str(), 2*M_PI/1000., 512*M_PI/1000., 2*M_PI/1000., 1.0, 1.0);

	// Integrated Bispectrum
	stat.compute_IntegratedBiSpec(fout+fend.str()+".dat", 16*M_PI/1500., 512*M_PI/1500., 8*M_PI/1500., 8);

	// =============================================================================================
      }
    }

  MPI_Finalize();
  return 0;
}
