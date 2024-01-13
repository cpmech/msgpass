#include <inttypes.h>
#include <stdlib.h>

#include "mpi.h"

#include "constants.h"

// https://www.open-mpi.org/doc/v4.1/

int32_t c_mpi_init() {
    int status = MPI_Init(NULL, NULL); // initializes the MPI execution environment
    return status;
}

int32_t c_mpi_init_threaded() {
    int provided;
    int status = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided); // initializes the MPI execution environment
    if (provided != MPI_THREAD_MULTIPLE) {
        return C_MPI_ERROR_INIT_THREADED;
    }
    return status;
}

int32_t c_mpi_finalize() {
    int status = MPI_Finalize(); // terminates MPI execution environment
    return status;
}

int32_t c_mpi_initialized(int32_t *flag) {
    int status = MPI_Initialized(flag); // checks whether MPI has been initialized or not
    return status;
}

int32_t c_mpi_world_rank(int32_t *rank) {
    int status = MPI_Comm_rank(MPI_COMM_WORLD, rank); // determines the rank of the calling process in the communicator
    return status;
}

int32_t c_mpi_world_size(int32_t *size) {
    int status = MPI_Comm_size(MPI_COMM_WORLD, size); // returns the size of the group associated with a communicator
    return status;
}

struct ExtCommunicator {
    MPI_Comm handle;
    MPI_Group group;
};

void comm_drop(struct ExtCommunicator *comm) {
    if (comm != NULL) {
        free(comm);
    }
}

struct ExtCommunicator *comm_new() {
    struct ExtCommunicator *comm = (struct ExtCommunicator *)malloc(sizeof(struct ExtCommunicator));
    if (comm == NULL) {
        return NULL;
    }

    comm->handle = MPI_COMM_WORLD;
    int status = MPI_Comm_group(MPI_COMM_WORLD, &comm->group); // returns the group associated with a communicator
    if (status != MPI_SUCCESS) {
        free(comm);
        return NULL;
    }

    return comm;
}

struct ExtCommunicator *comm_new_subset(int32_t n_rank, int32_t const *ranks) {
    struct ExtCommunicator *comm = (struct ExtCommunicator *)malloc(sizeof(struct ExtCommunicator));
    if (comm == NULL) {
        return NULL;
    }

    MPI_Group world_group;
    int status = MPI_Comm_group(MPI_COMM_WORLD, &world_group); // returns the group associated with a communicator
    if (status != MPI_SUCCESS) {
        free(comm);
        return NULL;
    }

    status = MPI_Group_incl(world_group, n_rank, ranks, &comm->group); // produces a group by reordering an existing group and taking only listed members
    if (status != MPI_SUCCESS) {
        free(comm);
        return NULL;
    }

    status = MPI_Comm_create(MPI_COMM_WORLD, comm->group, &comm->handle); // creates a new communicator
    if (status != MPI_SUCCESS) {
        free(comm);
        return NULL;
    }

    return comm;
}

int32_t comm_abort(struct ExtCommunicator *comm, int32_t error_code) {
    int status = MPI_Abort(comm->handle, error_code); // Terminates MPI execution environment
    return status;
}

int32_t comm_broadcast(struct ExtCommunicator *comm, int32_t sender, int32_t n, void *x, int32_t type_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Bcast(x, n, dty, sender, comm->handle); // broadcasts a message from the process with rank root to all other processes of the group
    return status;
}

int32_t comm_reduce(struct ExtCommunicator *comm, int32_t root, int32_t n, void *dest, void const *orig, int32_t type_index, int32_t op_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    MPI_Op op = C_MPI_OPS[op_index];
    int status = MPI_Reduce(orig, dest, n, dty, op, root, comm->handle); // reduces values on all processes within a group
    return status;
}

int32_t comm_allreduce(struct ExtCommunicator *comm, int32_t n, void *dest, void const *orig, int32_t type_index, int32_t op_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    MPI_Op op = C_MPI_OPS[op_index];
    int status = MPI_Allreduce(orig, dest, n, dty, op, comm->handle); // combines values from all processes and distributes the result back to all processes
    return status;
}
