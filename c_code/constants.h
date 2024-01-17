#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <inttypes.h>

#include "mpi.h"

const int32_t C_MPI_ERROR_INIT_THREADED = 10000;

const int C_MPI_THREAD_OPTIONS[4] = {
    MPI_THREAD_SINGLE,     //  0  only one thread will execute
    MPI_THREAD_FUNNELED,   //  1  if the process is multithreaded, only the thread that called MPI_Init_thread will make MPI calls
    MPI_THREAD_SERIALIZED, //  2  if the process is multithreaded, only one thread will make MPI library calls at one time
    MPI_THREAD_MULTIPLE,   //  3  if the process is multithreaded, multiple threads may call MPI at once with no restrictions
};

const MPI_Datatype C_MPI_TYPES[8] = {
    MPI_INT32_T,          //  0  i32
    MPI_INT64_T,          //  1  i64
    MPI_UINT32_T,         //  2  u32
    MPI_UINT64_T,         //  3  u64
    MPI_FLOAT,            //  4  f32
    MPI_DOUBLE,           //  5  f64
    MPI_C_FLOAT_COMPLEX,  //  6  c32
    MPI_C_DOUBLE_COMPLEX, //  7  c64
};

const MPI_Op C_MPI_OPS[7] = {
    MPI_MAX,  //  0  maximum
    MPI_MIN,  //  1  minimum
    MPI_SUM,  //  2  sum
    MPI_PROD, //  3  product
    MPI_LAND, //  4  logical and
    MPI_LOR,  //  5  logical or
    MPI_LXOR, //  6  logical xor
};

#endif // CONSTANTS_H
