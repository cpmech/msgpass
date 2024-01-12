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

int c_mpi_world_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int c_mpi_world_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}
