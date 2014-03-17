#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
#include <vector>

#define NUM_THREADS 256

using namespace std;

extern double size;
//
//  benchmarking program
//



__device__ double bin_size;
double host_bin_size;
__device__ int bins_per_side;
int host_bins_per_side;


vector<vector<int> > bins;
int* host_ptr_bin_list; 
int* host_ptr_bin_index; 
__device__ int* bin_list;
__device__ int* bin_index; 




#define BIN_SIZE bin_size
#define NUM_BINS_PER_SIDE bins_per_side
#define NUM_BINS (NUM_BINS_PER_SIDE*NUM_BINS_PER_SIDE)

#define HOST_BIN_SIZE host_bin_size
#define HOST_NUM_BINS_PER_SIDE host_bins_per_side
#define HOST_NUM_BINS (HOST_NUM_BINS_PER_SIDE*HOST_NUM_BINS_PER_SIDE)





void set_device_params(double local_bin_size, int local_bins_per_side) 
{
    host_bin_size = local_bin_size;
    host_bins_per_side = local_bins_per_side;

    cudaMemcpyToSymbol(bins_per_side, &local_bins_per_side, sizeof(int));
    cudaMemcpyToSymbol(bin_size, &local_bin_size, sizeof(double));
}






void set_bin_size( int n )
{
    double local_bin_size = sqrt(density*5);
    int local_bins_per_side = (int)(floor(size/local_bin_size)+1);

    set_device_params(local_bin_size, local_bins_per_side);
}





int which_bin(particle_t &particle) 
{
    int col = (int)floor(particle.x/HOST_BIN_SIZE);
    int row = (int)floor(particle.y/HOST_BIN_SIZE);
    return row * HOST_NUM_BINS_PER_SIDE + col;
}





__global__ void ptr_init(int* host_ptr_bin_list, int* host_ptr_bin_index, int n)
{
    bin_list = host_ptr_bin_list;
    bin_index = host_ptr_bin_index;

    for (int i = 0; i < n; i++) 
    {
        bin_list[i] = -1;
    }
    for(int i = 0; i < NUM_BINS; i++) {
        bin_index[i] = -1;
    }
}





void bin_init(int n) 
{
    bins.resize(HOST_NUM_BINS);
    host_ptr_bin_list = NULL;
    host_ptr_bin_index = NULL;
    cudaMalloc((void**)&host_ptr_bin_list, n * sizeof(int));
    cudaMalloc((void**)&host_ptr_bin_index, HOST_NUM_BINS * sizeof(int));
    ptr_init<<<1,1>>>(host_ptr_bin_list, host_ptr_bin_index, n);
}






void bin_particles(particle_t* particles, int n) 
{
    for(int i = 0; i < HOST_NUM_BINS; i++) 
    {
        bins[i].clear();
    }
    for(int i = 0; i < n; i++) 
    {
        int bin = which_bin(particles[i]);
        bins[bin].push_back(i);
    }
    int accum = 0;
    for(int i = 0; i < HOST_NUM_BINS; i++) 
    {
        int size = bins[i].size();
        cudaMemcpy(&(host_ptr_bin_list[accum]), &(bins[i])[0], size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&(host_ptr_bin_index[i]), &accum, sizeof(int), cudaMemcpyHostToDevice);
        accum += size;
    }
}






__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}






__device__ void neighbor_compare(int index, int n, int p, particle_t * particles) 
{
    const int start = bin_index[index]; 
    const int end = (index)+1 < NUM_BINS ? bin_index[(index)+1] : n; 
    for(int idx2 = start; idx2 < end; ++idx2) 
    { 
        apply_force_gpu( particles[p], particles[bin_list[idx2]] ); 
    } 
}






__device__ void compute_forces_for_box(particle_t * particles, int n, int r, int c) {
  const int i = r * NUM_BINS_PER_SIDE + c;

  const int start = bin_index[i];
  const int end = i+1 < NUM_BINS ? bin_index[i+1] : n;

  for(int idx=start; idx<end; ++idx) 
  {
    int p = bin_list[idx]; 
    particles[p].ax = 0;
    particles[p].ay = 0;
    
    neighbor_compare(i, n, p, particles);

    if( c-1 >= 0 )
    {
        neighbor_compare(i-1, n, p, particles);
    }
    if( c+1 < NUM_BINS_PER_SIDE )
    {
        neighbor_compare(i+1, n, p, particles);
    }
    if (r-1 >= 0)
    {
        neighbor_compare(i-NUM_BINS_PER_SIDE, n, p, particles);
    }
    if (r+1 < NUM_BINS_PER_SIDE)
    {
        neighbor_compare(i+NUM_BINS_PER_SIDE, n, p, particles);
    }


    if ((c - 1 >= 0) && (r-1 >= 0))
    {
        neighbor_compare(i-NUM_BINS_PER_SIDE-1, n, p, particles);
    }
    if ((c + 1 < NUM_BINS_PER_SIDE) && (r-1 >= 0))
    {
        neighbor_compare(i-NUM_BINS_PER_SIDE+1, n, p, particles);
    }
    if ((c - 1 >= 0) && (r+1 < NUM_BINS_PER_SIDE))
    {
        neighbor_compare(i+NUM_BINS_PER_SIDE-1, n, p, particles);
    }
    if ((c + 1 < NUM_BINS_PER_SIDE) && (r+1 < NUM_BINS_PER_SIDE))
    {
        neighbor_compare(i+NUM_BINS_PER_SIDE+1, n, p, particles);
    }
  }
}










__global__ void compute_forces_gpu(particle_t * particles, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= NUM_BINS) return;

  int i = tid;
  int r = i/NUM_BINS_PER_SIDE;
  int c = i % NUM_BINS_PER_SIDE;
  compute_forces_for_box(particles, n, r, c);
}







__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}





int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );
    set_bin_size( n );

    init_particles( n, particles );

    bin_init(n);
    bin_particles(particles, n);

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //

	int blks = (HOST_NUM_BINS + NUM_THREADS - 1) / NUM_THREADS;
	compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);
        
        //
        //  move particles
        //
    int move_blks = (n + NUM_THREADS - 1) / NUM_THREADS;
	move_gpu <<< move_blks, NUM_THREADS >>> (d_particles, n, size);
        
        //
        //  save if necessary
        //
    cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    bin_particles(particles, n);  

        if( fsave && (step%SAVEFREQ) == 0 ) 
        {
	    // Copy the particles back to the CPU
            save( fsave, n, particles);
        }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
