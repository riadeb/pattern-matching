/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda_runtime.h>


#define APM_DEBUG 0

char * 
read_input_file( char * filename, int * size )
{
    char * buf ;
    off_t fsize;
    int fd = 0 ;
    int n_bytes = 1 ;

    /* Open the text file */
    fd = open( filename, O_RDONLY ) ;
    if ( fd == -1 ) 
    {
        fprintf( stderr, "Unable to open the text file <%s>\n", filename ) ;
        return NULL ;
    }


    /* Get the number of characters in the textfile */
    fsize = lseek(fd, 0, SEEK_END);
    if ( fsize == -1 )
    {
        fprintf( stderr, "Unable to lseek to the end\n" ) ;
        return NULL ;
    }

#if APM_DEBUG
    printf( "File length: %lld\n", fsize ) ;
#endif

    /* Go back to the beginning of the input file */
    if ( lseek(fd, 0, SEEK_SET) == -1 ) 
    {
        fprintf( stderr, "Unable to lseek to start\n" ) ;
        return NULL ;
    }

    /* Allocate data to copy the target text */
    buf = (char *)malloc( fsize * sizeof ( char ) ) ;
    if ( buf == NULL ) 
    {
        fprintf( stderr, "Unable to allocate %lld byte(s) for main array\n",
                fsize ) ;
        return NULL ;
    }

    n_bytes = read( fd, buf, fsize ) ;
    if ( n_bytes != fsize ) 
    {
        fprintf( stderr, 
                "Unable to copy %lld byte(s) from text file (%d byte(s) copied)\n",
                fsize, n_bytes) ;
        return NULL ;
    }

#if APM_DEBUG
    printf( "Number of read bytes: %d\n", n_bytes ) ;
#endif

    *size = n_bytes ;


    close( fd ) ;


    return buf ;
}


#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

 int levenshtein(char *s1, char *s2, int len, int * column) {
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
                    lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1)
                    );
            lastdiag = olddiag;

        }
    }
    return(column[len]);
}

__global__ void compMatches(char* pattern,char * buf, int n_bytes, int size_pattern, int approx_factor, int * resultsth){
         int distance = 0 ;
          int size ;
          int i = threadIdx.x + blockIdx.x*blockDim.x;
          int n_th = gridDim.x*blockDim.x;
          resultsth[i] = 0;
          int * column = (int *)malloc( (size_pattern+1) * sizeof( int ) ) ;
            if(i < n_bytes){
                for(int j = i; j < n_bytes; j += n_th){
                    size = size_pattern ;
                    if ( n_bytes - j < size_pattern )
                    {
                        size = n_bytes - j ;
                    }
    
                    char * s1 = pattern;
                    char * s2 = &buf[j];
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
                                    lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1)
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

void checkGpuMem()

{

float free_m,total_m,used_m;

size_t free_t,total_t;

cudaMemGetInfo(&free_t,&total_t);

free_m =(uint)free_t/1048576.0 ;

total_m=(uint)total_t/1048576.0;

used_m=total_m-free_m;

printf ( "  mem free %d .... %f MB mem total %d....%f MB mem used %f MB\n",free_t,free_m,total_t,total_m,used_m);

}

int 
main( int argc, char ** argv )
{
  char ** pattern ;
  char * filename ;
  int approx_factor = 0 ;
  int nb_patterns = 0 ;
  char * buf ;
  struct timeval t1, t2;
  double duration ;
  int n_bytes ;
  int * n_matches ;

  /* Check number of arguments */
  if ( argc < 4 ) 
  {
    printf( "Usage: %s approximation_factor "
            "dna_database pattern1 pattern2 ...\n", 
            argv[0] ) ;
    return 1 ;
  }

  /* Get the distance factor */
  approx_factor = atoi( argv[1] ) ;

  /* Grab the filename containing the target text */
  filename = argv[2] ;

  /* Get the number of patterns that the user wants to search for */
  nb_patterns = argc - 3 ;

  /* Fill the pattern array */
  pattern = (char **)malloc( nb_patterns * sizeof( char * ) ) ;
  if ( pattern == NULL ) 
  {
      fprintf( stderr, 
              "Unable to allocate array of pattern of size %d\n", 
              nb_patterns ) ;
      return 1 ;
  }
  checkGpuMem();

  /* Grab the patterns */
  for (int i = 0 ; i < nb_patterns ; i++ ) 
  {
      int l ;

      l = strlen(argv[i+3]) ;
      if ( l <= 0 ) 
      {
          fprintf( stderr, "Error while parsing argument %d\n", i+3 ) ;
          return 1 ;
      }

      pattern[i] = (char *)malloc( (l+1) * sizeof( char ) ) ;
      if ( pattern[i] == NULL ) 
      {
          fprintf( stderr, "Unable to allocate string of size %d\n", l ) ;
          return 1 ;
      }

      strncpy( pattern[i], argv[i+3], (l+1) ) ;
  }


  printf( "Approximate Pattern Mathing: "
          "looking for %d pattern(s) in file %s w/ distance of %d\n", 
          nb_patterns, filename, approx_factor ) ;

  buf = read_input_file( filename, &n_bytes ) ;
  if ( buf == NULL )
  {
      return 1 ;
  }
  cudaError_t err;
    char * cbuf;
    err = cudaMalloc((void **)&cbuf, n_bytes* sizeof(char));
    if( err != cudaSuccess) {
        printf("Error !");
        exit(1);
    }
    err = cudaMemcpy(cbuf, buf,n_bytes* sizeof(char) ,cudaMemcpyHostToDevice);
        if( err != cudaSuccess) {
            printf("Error !");
            exit(1);
        }
    int nblock = 100;
    int nth_b = 1024;
    int nth = nblock*nth_b;

    int * results_th;
    err = cudaMalloc((void **)&results_th, nth* sizeof(int));
    if( err != cudaSuccess) {
        printf("Error !");
        exit(1);
    }



  /* Allocate the array of matches */
  n_matches = (int *)malloc( nb_patterns * sizeof( int ) ) ;
  if ( n_matches == NULL )
  {
      fprintf( stderr, "Error: unable to allocate memory for %ldB\n",
              nb_patterns * sizeof( int ) ) ;
      return 1 ;
  }

  /*****
   * BEGIN MAIN LOOP
   ******/

  /* Timer start */
  gettimeofday(&t1, NULL);

  /* Check each pattern one by one */
  for (int i = 0 ; i < nb_patterns ; i++ )
  {
      int size_pattern = strlen(pattern[i]) ;

      /* Initialize the number of matches to 0 */
      n_matches[i] = 0 ;

      char * cpattern;
      err = cudaMalloc((void **)&cpattern, size_pattern* sizeof(char));
    if( err != cudaSuccess) {
        printf("Error !");
        exit(1);
    }
        err= cudaMemcpy(cpattern,pattern[i], size_pattern* sizeof(char), cudaMemcpyHostToDevice);
        if( err != cudaSuccess) {
            printf("Error !");
            exit(1);
        }


      compMatches<<<nblock, nth_b>>>(cpattern,cbuf,n_bytes,size_pattern,approx_factor,results_th);
      err = cudaGetLastError();
      if( err != cudaSuccess) {
        printf("Error ! in kernel");
        exit(1);
    }  
      int * results;
        results = (int *)malloc(nth* sizeof(int));
        err = cudaMemcpy(results,results_th, nth* sizeof(int), cudaMemcpyDeviceToHost);
        if( err != cudaSuccess) {
            printf("Error ! in copying results %s", cudaGetErrorString(err));
            exit(1);
        }  
        for(int j = 0; j < nth && j < n_bytes; j++){
            n_matches[i] += results[j];
        }
  }

  /* Timer stop */
  gettimeofday(&t2, NULL);

  duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

  printf( "APM done in %lf s\n", duration ) ;

  /*****
   * END MAIN LOOP
   ******/

  for ( int i = 0 ; i < nb_patterns ; i++ )
  {
      printf( "Number of matches for pattern <%s>: %d\n", 
              pattern[i], n_matches[i] ) ;
  }

  return 0 ;
}
