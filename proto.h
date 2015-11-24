#ifndef PROTO_H
#define PROTO_H

struct io_header_1
{
  int      npart[6];
  double   mass[6];
  double   time;
  double   redshift;
  int      flag_sfr;
  int      flag_feedback;
  int      npartTotal[6];
  int      flag_cooling;
  int      num_files;
  double   BoxSize;
  double   Omega0;
  double   OmegaLambda;
  double   HubbleParam; 
  char     fill[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 4*8];  /* fills to 256 Bytes */
};

struct particle_data 
{
  float  Pos[3];
  float  Vel[3];
  float  Mass;
  int    Type;

  float  Rho, U, Temp, Ne;
};

struct particle_data_pos
{
  float Pos[3];
};


int load_snapshot(char *fname, int files, 
		  struct io_header_1 header1, struct particle_data_pos *P);

struct particle_data_pos* load_sub_snapshot(char *fname, int file, int &nPart_file);

int write_snapshot(char *fname, int NEWfiles);

int get_header_info(void);

int unit_conversion(void);

size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream);

// double HubbleParam;
// double Omega0;
// double OmegaLambda;
// double BoxSize;
// double atime;
// double redshift;
  
// int npart[6];  
// int npartTotal[6];  
// int nfiles;


struct particle_data_pos* allocate_memory();
void deallocate_memory(struct particle_data_pos *P);
void reordering(struct particle_data_pos *& P);

void do_what_you_want(void);

#endif
