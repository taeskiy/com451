/*******************************************************************************
*
*   probe a comuter for basic info about processing cores and GPU
*
*   compile with:
*
*			nvcc probe2.cu -L/usr/local/cuda/lib64 -I/usr/local/cuda-10.2/targets/x86_64-linux/include -lcuda -lcudart
*
*			(in .tcshrc, please have:)
*			set path = ($path /usr/local/cuda-10.1/bin ./)
*			setenv LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda-10.1/lib64
*
*******************************************************************************/
#include <stdio.h>

void getInfo(){

  cudaDeviceProp prop;

    printf("\tName: %s\n", prop.name);
    printf("\tTotal global mem: %ld\n", prop.totalGlobalMem );
    printf( "\tShared mem per processor: %ld\n", prop.sharedMemPerBlock );
    printf( "\tMax threads per block: %d\n", prop.maxThreadsPerBlock );
    printf( "\tMax block dimensions: (%d, %d, %d)\n", 
                  prop.maxThreadsDim[0],
                  prop.maxThreadsDim[1],
                  prop.maxThreadsDim[2]);
    printf("\tTotal constant Mem: %ld\n", prop.totalConstMem );
    printf( "\tMultiprocessor count: %d\n", prop.multiProcessorCount ); 

    printf("\n");



/******************************************************************************/
int main(int argc, char *argv[]){

getInfo();

return 0;
}

/******************************************************************************/
