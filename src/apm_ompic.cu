/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
 #include <stdio.h>
 #include <string.h>
 #include <stdlib.h>
 #include <fcntl.h>
 #include <unistd.h>
 #include <sys/time.h>


 #include <cuda_runtime.h>
 #include <cuda.h>

 #define APM_DEBUG 0

 #define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
 #define MIN(a, b) ((a) < (b) ? (a) : (b) )

int * results_th ;

 void CHECK(cudaError_t err){
    if( err != cudaSuccess) {
        printf("Error ! %s \n", cudaGetErrorString(err));
        exit(1);
    }  
 }

__global__ void compMatches(char* pattern,char * buf,int cuda_end ,int n_bytes, int size_pattern, int approx_factor, int * resultsth){
   
    int distance = 0 ;
     int size ;
     int i = threadIdx.x + blockIdx.x*blockDim.x;
     int n_th = gridDim.x*blockDim.x;
     resultsth[i] = 0;
     if(i == 0)     printf("cuda started \n");
       if(i < cuda_end){
        int * column = (int *)malloc( (size_pattern+1) * sizeof( int ) ) ;
           for(int j = i; j < cuda_end; j += n_th){
               size = size_pattern ;
               if ( n_bytes - j < size_pattern )
               {
                   size = n_bytes - j ;
               }
               int len = size;
               unsigned int x, y, lastdiag, olddiag;

               for (y = 1; y <= len; y++)
               {
                   column[y] = y;
               }
               for (x = 1; x <= len; x++) {
                   column[0] = x;
                   lastdiag = x-1 ;
                   for (y = 1; y <= len; y++) {
                       olddiag = column[y];
                       column[y] = MIN3(
                               column[y] + 1, 
                               column[y-1] + 1, 
                               lastdiag + (pattern[y-1] == buf[j+x-1] ? 0 : 1)
                               );
                       lastdiag = olddiag;

                   }
               }
               distance = column[len];
               if ( distance <= approx_factor ) {
                   resultsth[i]++ ;
               }
         }
       }
}
extern "C" void kernelCall(char* cpattern,char * cbuf,int cuda_end ,int n_bytes, int size_pattern, int approx_factor, int * results_th , int  nth_b,int nblock){

    compMatches<<<nblock, nth_b>>>(cpattern,cbuf,cuda_end,n_bytes,size_pattern,approx_factor,results_th);
    CHECK(cudaGetLastError());
}
extern "C" int  finalcudaCall(char* cpattern,char * cbuf, int cuda_end, int * results_th , int  nth_b,int nblock){
    int * results;
    int nth = nth_b*nblock;
    results = (int *)malloc(nth* sizeof(int));
    CHECK(cudaDeviceSynchronize());
   printf("cuda done\n");
    CHECK(cudaMemcpy(results,results_th, nth* sizeof(int), cudaMemcpyDeviceToHost));

    int res = 0;
    for(int j = 0; j < nth && j < cuda_end; j++){
        res += results[j];
    }
    return res;
}

 extern "C" char * cuda_malloc_cp(char *buf, int size){
    char *dBuf;
    CHECK(cudaMalloc( (void**)&dBuf, size));
    CHECK(cudaMemcpy(dBuf, buf, size, cudaMemcpyHostToDevice) );
    return dBuf;
}
extern "C" int * cuda_malloc(int size){
    int *dBuf;
    CHECK(cudaMalloc( (void **)&dBuf, size));
    return dBuf;
}