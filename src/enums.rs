/// Specifies the option for MPI_Init_thread
///
/// From <https://enccs.github.io/intermediate-mpi/mpi-and-threads-pt1/>
///
/// * MPI_THREAD_SINGLE - rank is not allowed to use threads, which is basically equivalent to calling MPI_Init.
///   With MPI_THREAD_SINGLE, the rank may use MPI freely and will not use threads.
/// * MPI_THREAD_FUNNELED - rank can be multi-threaded but only the main thread may call MPI functions.
///   Ideal for fork-join parallelism such as used in #pragma omp parallel, where all MPI calls are outside the OpenMP regions.
///   With MPI_THREAD_FUNNELED, the rank can use MPI from only the main thread.
/// * MPI_THREAD_SERIALIZED - rank can be multi-threaded but only one thread at a time may call MPI functions.
///   The rank must ensure that MPI is used in a thread-safe way.
///   One approach is to ensure that MPI usage is mutually excluded by all the threads, eg. with a mutex.
///   With MPI_THREAD_SERIALIZED, the rank can use MPI from any thread so long as it ensures the threads synchronize
///   such that no thread calls MPI while another thread is doing so.
/// * MPI_THREAD_MULTIPLE - rank can be multi-threaded and any thread may call MPI functions.
///   The MPI library ensures that this access is safe across threads.
///   Note that this makes all MPI operations less efficient, even if only one thread makes MPI calls,
///   so should be used only where necessary. With MPI_THREAD_MULTIPLE, the rank can use MPI from any thread.
///   The MPI library ensures the necessary synchronization
///
/// **Note:** The performance may be affected with MPI_THREAD_MULTIPLE
#[derive(Clone, Copy)]
pub enum MpiThread {
    /// Only one thread will execute
    Single = 0,

    /// The process may be multi-threaded, but only the main thread will make MPI calls (all MPI calls are funneled to the main thread)
    Funneled = 1,

    /// The process may be multi-threaded, and multiple threads may make MPI calls, but only one at a time: MPI calls are not made concurrently from two distinct threads (all MPI calls are serialized)
    Serialized = 2,

    /// Multiple threads may call MPI, with no restrictions
    Multiple = 3,
}

#[derive(Clone, Copy)]
pub(crate) enum MpiType {
    I32 = 0,
    I64 = 1,
    U32 = 2,
    U64 = 3,
    F32 = 4,
    F64 = 5,
    C32 = 6, // Complex32
    C64 = 7, // Complex64
    BYT = 8, // u8 (Byte)
}

/// Specifies the MPI operator used in reduce-like functions (for integer arrays)
#[derive(Clone, Copy)]
pub enum MpiOpInt {
    Max = 0,  // maximum
    Min = 1,  // minimum
    Sum = 2,  // sum
    Prod = 3, // product
    Land = 4, // logical and
    Lor = 5,  // logical or
    Lxor = 6, // logical xor
}

/// Specifies the MPI operator used in reduce-like functions (for real number arrays)
#[derive(Clone, Copy)]
pub enum MpiOpReal {
    Max = 0,  // maximum
    Min = 1,  // minimum
    Sum = 2,  // sum
    Prod = 3, // product
}

/// Specifies the MPI operator used in reduce-like functions (for complex number arrays)
#[derive(Clone, Copy)]
pub enum MpiOpComplex {
    Sum = 2,  // sum
    Prod = 3, // product
}

impl MpiThread {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiType {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiOpInt {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiOpReal {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiOpComplex {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}
