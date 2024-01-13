#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <inttypes.h>

#include "mpi.h"

const int32_t C_MPI_ERROR_INIT_THREADED = 10000;

const MPI_Datatype C_MPI_TYPES[6] = {
    MPI_INT32_T,  //  0  i32
    MPI_INT64_T,  //  1  i64
    MPI_UINT32_T, //  2  u32
    MPI_UINT64_T, //  3  u64
    MPI_FLOAT,    //  4  f32
    MPI_DOUBLE,   //  5  f64
};

const MPI_Op C_MPI_OPS[10] = {
    MPI_MAX,  //  0  maximum
    MPI_MIN,  //  1  minimum
    MPI_SUM,  //  2  sum
    MPI_PROD, //  3  product
    MPI_LAND, //  4  logical and
    MPI_BAND, //  5  bit-wise and
    MPI_LOR,  //  6  logical or
    MPI_BOR,  //  7  bit-wise or
    MPI_LXOR, //  8  logical xor
    MPI_BXOR, //  9  bit-wise xor
};

#endif // CONSTANTS_H
