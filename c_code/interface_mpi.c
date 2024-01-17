#include <inttypes.h>
#include <stdlib.h>

#include "mpi.h"

#include "constants.h"

// References:
// https://www.open-mpi.org/doc/v4.1/
// https://rookiehpc.org/mpi/docs/mpi_gather/index.html
// https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/

int32_t c_mpi_init() {
    int status = MPI_Init(NULL, NULL); // initializes the MPI execution environment
    return status;
}

int32_t c_mpi_init_thread(int32_t option_index) {
    int option = C_MPI_THREAD_OPTIONS[option_index];
    int provided;
    int status = MPI_Init_thread(NULL, NULL, option, &provided); // initializes the MPI execution environment
    if (provided != option) {
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
    MPI_Status recv_status;
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

    comm->recv_status.MPI_SOURCE = 0;
    comm->recv_status.MPI_TAG = 0;
    comm->recv_status.MPI_ERROR = 0;

    return comm;
}

int32_t comm_abort(struct ExtCommunicator *comm, int32_t error_code) {
    int status = MPI_Abort(comm->handle, error_code); // terminates MPI execution environment
    return status;
}

int32_t comm_barrier(struct ExtCommunicator *comm) {
    int status = MPI_Barrier(comm->handle); // synchronization between MPI processes
    return status;
}

int32_t comm_rank(struct ExtCommunicator *comm, int32_t *rank) {
    int status = MPI_Comm_rank(comm->handle, rank); // determines the rank of the calling process in the communicator
    return status;
}

int32_t comm_size(struct ExtCommunicator *comm, int32_t *size) {
    int status = MPI_Comm_size(comm->handle, size); // returns the size of the group associated with a communicator
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

int32_t comm_send(struct ExtCommunicator *comm, int32_t n, void *data, int32_t type_index, int32_t to_rank, int32_t tag) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Send(data, n, dty, to_rank, tag, comm->handle); // performs a standard-mode blocking send
    return status;
}

// from_rank < 0 corresponds to MPI_ANY_SOURCE
// tag < 0 corresponds to MPI_ANY_TAG
int32_t comm_receive(struct ExtCommunicator *comm, int32_t n, void *data, int32_t type_index, int32_t from_rank, int32_t tag) {
    int r = from_rank < 0 ? MPI_ANY_SOURCE : from_rank;
    int t = tag < 0 ? MPI_ANY_TAG : tag;
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    comm->recv_status.MPI_SOURCE = r;
    comm->recv_status.MPI_TAG = t;
    comm->recv_status.MPI_ERROR = MPI_SUCCESS;
    int status = MPI_Recv(data, n, dty, r, t, comm->handle, &comm->recv_status); // performs a standard-mode blocking receive
    return status;
}

void comm_get_receive_status(struct ExtCommunicator *comm, int32_t *source, int32_t *tag, int32_t *error) {
    *source = comm->recv_status.MPI_SOURCE;
    *tag = comm->recv_status.MPI_TAG;
    *error = comm->recv_status.MPI_ERROR;
}

// len(dest) must be equal to n * n_processors
// len(orig) must be equal to n
int32_t comm_gather_im_root(struct ExtCommunicator *comm, int32_t root, int32_t n, void *dest, void const *orig, int32_t type_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Gather(orig, n, dty, dest, n, dty, root, comm->handle); // gathers values from a group of processes.
    return status;
}

int32_t comm_gather_im_not_root(struct ExtCommunicator *comm, int32_t root, int32_t n, void const *orig, int32_t type_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Gather(orig, n, dty, NULL, 0, dty, root, comm->handle); // gathers values from a group of processes.
    return status;
}

// len(dest) must be equal to n * n_processors
// len(orig) must be equal to n
int32_t comm_allgather(struct ExtCommunicator *comm, int32_t n, void *dest, void const *orig, int32_t type_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Allgather(orig, n, dty, dest, n, dty, comm->handle); // gathers data from all processes
    return status;
}

// len(dest) must be equal to n
// len(orig) must be equal to n * n_processors
int32_t comm_scatter_im_root(struct ExtCommunicator *comm, int32_t root, int32_t n, void *dest, void const *orig, int32_t type_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Scatter(orig, n, dty, dest, n, dty, root, comm->handle); // sends data from one task to all tasks in a group
    return status;
}

// len(dest) must be equal to n
int32_t comm_scatter_im_not_root(struct ExtCommunicator *comm, int32_t root, int32_t n, void *dest, int32_t type_index) {
    MPI_Datatype dty = C_MPI_TYPES[type_index];
    int status = MPI_Scatter(NULL, 0, dty, dest, n, dty, root, comm->handle); // sends data from one task to all tasks in a group
    return status;
}
