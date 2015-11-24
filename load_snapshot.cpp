#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <iostream>

#include "proto.h"



int NumPart, Ngas;
int *Id;
double  Time, Redshift;


/* this routine loads particle data from Gadget's default
 * binary file format. (A snapshot may be distributed
 * into multiple files.
 */
int load_snapshot(char *fname, int files, struct io_header_1 header1, struct particle_data_pos *P)
{
  FILE *fd;
  char   buf[200];
  int    i,j,k,dummy,ntot_withmasses;
  int    t,n,off,pc,pc_new,pc_sph;

#define SKIP fread(&dummy, sizeof(dummy), 1, fd);

  for(i=0, pc=1; i<files; i++, pc=pc_new)
    {
      if(files>1)
	sprintf(buf,"%s.%d",fname,i);
      else
	sprintf(buf,"%s",fname);

      if(!(fd=fopen(buf,"r")))
	{
	  printf("can't open file `%s`\n",buf);
	  exit(0);
	}

      printf("reading `%s' ...\n",buf); fflush(stdout);

      fread(&dummy, sizeof(dummy), 1, fd);
      fread(&header1, sizeof(header1), 1, fd);
      fread(&dummy, sizeof(dummy), 1, fd);

      if(files==1)
	{
	  for(k=0, NumPart=0, ntot_withmasses=0; k<5; k++)
	    NumPart+= header1.npart[k];
	  Ngas= header1.npart[0];
	}
      else
	{
	  for(k=0, NumPart=0, ntot_withmasses=0; k<5; k++)
	    NumPart+= header1.npartTotal[k];
	  Ngas= header1.npartTotal[0];
	}

      for(k=0, ntot_withmasses=0; k<5; k++)
	{
	  if(header1.mass[k]==0)
	    ntot_withmasses+= header1.npart[k];
	}

      if(i==0)
	P = allocate_memory();

      SKIP;
      for(k=0,pc_new=pc;k<6;k++)
	{
	  for(n=0;n<header1.npart[k];n++)
	    {
	      fread(&P[pc_new].Pos[0], sizeof(float), 3, fd);
	      pc_new++;
	    }
	}
      SKIP;

      // SKIP;
      // for(k=0,pc_new=pc;k<6;k++)
      // 	{
      // 	  for(n=0;n<header1.npart[k];n++)
      // 	    {
      // 	      fread(&P[pc_new].Vel[0], sizeof(float), 3, fd);
      // 	      pc_new++;
      // 	    }
      // 	}
      // SKIP;

      // SKIP;
      // for(k=0,pc_new=pc;k<6;k++)
      // 	{
      // 	  for(n=0;n<header1.npart[k];n++)
      // 	    {
      // 	      fread(&Id[pc_new], sizeof(int), 1, fd);
      // 	      pc_new++;
      // 	    }
      // 	}
      // SKIP;


      // if(ntot_withmasses>0)
      // 	SKIP;
      // for(k=0, pc_new=pc; k<6; k++)
      // 	{
      // 	  for(n=0;n<header1.npart[k];n++)
      // 	    {
      // 	      P[pc_new].Type=k;

      // 	      if(header1.mass[k]==0)
      // 		fread(&P[pc_new].Mass, sizeof(float), 1, fd);
      // 	      else
      // 		P[pc_new].Mass= header1.mass[k];
      // 	      pc_new++;
      // 	    }
      // 	}
      // if(ntot_withmasses>0)
      // 	SKIP;
      

      // if(header1.npart[0]>0)
      // 	{
      // 	  SKIP;
      // 	  for(n=0, pc_sph=pc; n<header1.npart[0];n++)
      // 	    {
      // 	      fread(&P[pc_sph].U, sizeof(float), 1, fd);
      // 	      pc_sph++;
      // 	    }
      // 	  SKIP;

      // 	  SKIP;
      // 	  for(n=0, pc_sph=pc; n<header1.npart[0];n++)
      // 	    {
      // 	      fread(&P[pc_sph].Rho, sizeof(float), 1, fd);
      // 	      pc_sph++;
      // 	    }
      // 	  SKIP;

      // 	  if(header1.flag_cooling)
      // 	    {
      // 	      SKIP;
      // 	      for(n=0, pc_sph=pc; n<header1.npart[0];n++)
      // 		{
      // 		  fread(&P[pc_sph].Ne, sizeof(float), 1, fd);
      // 		  pc_sph++;
      // 		}
      // 	      SKIP;
      // 	    }
      // 	  else
      // 	    for(n=0, pc_sph=pc; n<header1.npart[0];n++)
      // 	      {
      // 		P[pc_sph].Ne= 1.0;
      // 		pc_sph++;
      // 	      }
      // 	}

      fclose(fd);
    }


  Time= header1.time;
  Redshift= header1.time;
  return NumPart;
}


/* this routine loads particle positions from one specified file
 * of a given snapshot.
 */
struct particle_data_pos* load_sub_snapshot(char *fname, int file, int &nPart_file)
{
  FILE *fd;
  char   buf[200];
  int    i,j,k,dummy,ntot_withmasses;
  int    t,n,off,pc,pc_new,pc_sph;
  struct particle_data_pos *P;
  struct io_header_1 header1;

#define SKIP fread(&dummy, sizeof(dummy), 1, fd);
  
  pc = 0;
  NumPart = 0.;

  sprintf(buf,"%s.%d",fname,file);

  if(!(fd=fopen(buf,"r")))
    {
      printf("can't open file `%s`\n",buf);
      exit(0);
    }

  printf("reading `%s' ...\n",buf); fflush(stdout);

  fread(&dummy, sizeof(dummy), 1, fd);
  fread(&header1, sizeof(header1), 1, fd);
  fread(&dummy, sizeof(dummy), 1, fd);

  for(k=0, NumPart=0, ntot_withmasses=0; k<5; k++)
    {
      NumPart+= header1.npart[k];
    }
  Ngas= header1.npart[0];

  nPart_file = NumPart;

  for(k=0, ntot_withmasses=0; k<5; k++)
    {
      if(header1.mass[k]==0)
	ntot_withmasses+= header1.npart[k];
    }

  // std::cout << "Number of particles (load_sub_snapshot): " << NumPart << std::endl;
  P = allocate_memory();

  SKIP;
  for(k=0,pc_new=pc;k<6;k++)
    {
      for(n=0;n<header1.npart[k];n++)
	{
	  fread(&(P[pc_new].Pos[0]), sizeof(float), 3, fd);
	  pc_new++;
	}
    }
  SKIP;

  fclose(fd);

  Time= header1.time;
  Redshift= header1.time;
  return P;
}


/* this routine allocates the memory for the 
 * particle data.
 */
struct particle_data_pos* allocate_memory()
{
  printf("allocating memory...");
  struct particle_data_pos *P;

  if(!(P=new(std::nothrow) particle_data_pos[NumPart]))
    {
      fprintf(stderr,"failed to allocate memory.\n");
      exit(0);
    }
  
  // P--;   /* start with offset 1 */
  
  // if(!(Id=new(std::nothrow) int[NumPart]))
  //   {
  //     fprintf(stderr,"failed to allocate memory.\n");
  //     exit(0);
  //   }
  
  // Id--;   /* start with offset 1 */
  
  printf("done\n");
  return P;
}


void deallocate_memory(struct particle_data_pos *P)
{
  free(P->Pos);
  free(P);
}




/* This routine brings the particles back into
 * the order of their ID's.
 * NOTE: The routine only works if the ID's cover
 * the range from 1 to NumPart !
 * In other cases, one has to use more general
 * sorting routines.
 */
void reordering(struct particle_data_pos *& P)
{
  int i,j;
  int idsource, idsave, dest;
  struct particle_data_pos psave, psource;


  printf("reordering...");

  for(i=1; i<=NumPart; i++)
    {
      if(Id[i] != i)
	{
	  psource= P[i];
	  idsource=Id[i];
	  dest=Id[i];

	  do
	    {
	      psave= P[dest];
	      idsave=Id[dest];

	      P[dest]= psource;
	      Id[dest]= idsource;
	      
	      if(dest == i) 
		break;

	      psource= psave;
	      idsource=idsave;

	      dest=idsource;
	    }
	  while(1);
	}
    }

  printf("done.\n");

  Id++;   
  free(Id);

  printf("space for particle ID freed\n");
}
