#include "memory.h"
#include "mpi.h"
#include "stdio.h"

void c_mpi_init() {
    MPI_Init(NULL, NULL);
}

void c_mpi_finalize() {
    MPI_Finalize();
}

int c_mpi_initialized() {
    int flag;
    MPI_Initialized(&flag);
    return flag;
}
