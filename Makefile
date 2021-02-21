SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
CFLAGS=-O3 -I$(HEADER_DIR) -Wall -std=c99 -fopenmp
LDFLAGS= 


all: $(OBJ_DIR) apm apm_mpi apm_omp apm_ompi

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^


apm:$(OBJ_DIR)/apm.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

apm_mpi: $(OBJ_DIR)/apm_mpi.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

apm_omp: $(OBJ_DIR)/apm_omp.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^	

apm_ompi: $(OBJ_DIR)/apm_ompi.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^	


clean:
	rm -f apm apm_mpi apm_omp $(OBJ) ; rm -r $(OBJ_DIR)
