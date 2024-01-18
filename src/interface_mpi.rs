use crate::constants::*;
use crate::conversion::to_i32;
use crate::enums::*;
use crate::StrError;
use num_complex::{Complex32, Complex64};
use std::ffi::c_void;

#[repr(C)]
struct ExtCommunicator {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn c_mpi_init() -> i32;
    fn c_mpi_init_thread(option_index: i32) -> i32;
    fn c_mpi_finalize() -> i32;
    fn c_mpi_initialized(flag: *mut i32) -> i32;
    fn c_mpi_world_rank(rank: *mut i32) -> i32;
    fn c_mpi_world_size(size: *mut i32) -> i32;
    fn comm_drop(comm: *mut ExtCommunicator);
    fn comm_new() -> *mut ExtCommunicator;
    fn comm_new_subset(n_rank: i32, ranks: *const i32) -> *mut ExtCommunicator;
    fn comm_abort(comm: *mut ExtCommunicator, error_code: i32) -> i32;
    fn comm_barrier(comm: *mut ExtCommunicator) -> i32;
    fn comm_rank(comm: *mut ExtCommunicator, rank: *mut i32) -> i32;
    fn comm_size(comm: *mut ExtCommunicator, size: *mut i32) -> i32;
    fn comm_broadcast(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut c_void, type_index: i32) -> i32;
    fn comm_reduce(comm: *mut ExtCommunicator, root: i32, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32, op_index: i32) -> i32;
    fn comm_allreduce(comm: *mut ExtCommunicator, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32, op_index: i32) -> i32;
    fn comm_send(comm: *mut ExtCommunicator, n: i32, data: *const c_void, type_index: i32, to_rank: i32, tag: i32) -> i32;
    fn comm_receive(comm: *mut ExtCommunicator, n: i32, data: *mut c_void, type_index: i32, from_rank: i32, tag: i32) -> i32;
    fn comm_get_receive_status(comm: *mut ExtCommunicator, source: *mut i32, tag: *mut i32, error: *mut i32);
    fn comm_gather_im_root(comm: *mut ExtCommunicator, root: i32, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32) -> i32;
    fn comm_gather_im_not_root(comm: *mut ExtCommunicator, root: i32, n: i32, orig: *const c_void, type_index: i32) -> i32;
    fn comm_allgather(comm: *mut ExtCommunicator, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32) -> i32;
    fn comm_scatter_im_root(comm: *mut ExtCommunicator, root: i32, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32) -> i32;
    fn comm_scatter_im_not_root(comm: *mut ExtCommunicator, root: i32, n: i32, dest: *mut c_void, type_index: i32) -> i32;
}

/// Initializes the MPI execution environment
pub fn mpi_init() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_init();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize");
        }
    }
    Ok(())
}

/// Initializes the MPI execution environment (with thread options)
///
/// See [MpiThread]
pub fn mpi_init_thread(option: MpiThread) -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_init_thread(option.n());
        if status == C_MPI_ERROR_INIT_THREADED {
            return Err("MPI failed to match the required thread option");
        }
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize (threaded)");
        }
    }
    Ok(())
}

/// Terminates the MPI execution environment
pub fn mpi_finalize() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_finalize();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to finalize");
        }
    }
    Ok(())
}

/// Checks whether MPI has been initialized or not
pub fn mpi_initialized() -> Result<bool, StrError> {
    unsafe {
        let mut flag: i32 = 0;
        let status = c_mpi_initialized(&mut flag);
        if status != C_MPI_SUCCESS {
            return Err("MPI cannot get initialized status");
        }
        Ok(if flag == 1 { true } else { false })
    }
}

/// Determines the rank of the calling process in the MPI_COMM_WORLD communicator
pub fn mpi_world_rank() -> Result<usize, StrError> {
    unsafe {
        let mut rank: i32 = 0;
        let status = c_mpi_world_rank(&mut rank);
        if status != C_MPI_SUCCESS {
            return Err("MPI cannot failed to get world rank");
        }
        Ok(rank as usize)
    }
}

/// Returns the size of the group associated with the MPI_COMM_WORLD communicator
pub fn mpi_world_size() -> Result<usize, StrError> {
    unsafe {
        let mut size: i32 = 0;
        let status = c_mpi_world_size(&mut size);
        if status != C_MPI_SUCCESS {
            return Err("MPI cannot failed to get world size");
        }
        Ok(size as usize)
    }
}

/// Implements the Rust communicator (wrapping the C data)
///
/// This struct holds a pointer to the C communicator, which stores the communicator (MPI_Comm),
/// the group (MPI_Group), and the status from recv calls (MPI_Status).
pub struct Communicator {
    handle: *mut ExtCommunicator,
}

impl Drop for Communicator {
    /// Deallocates the C memory
    fn drop(&mut self) {
        unsafe {
            comm_drop(self.handle);
        }
    }
}

impl Communicator {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let ext_comm = comm_new();
            if ext_comm.is_null() {
                return Err("MPI failed to return the world communicator");
            }
            Ok(Communicator { handle: ext_comm })
        }
    }

    /// Allocates a new instance using a subset of processors
    pub fn new_subset(ranks: &[usize]) -> Result<Self, StrError> {
        unsafe {
            let n = to_i32(ranks.len());
            let c_ranks: Vec<i32> = ranks.iter().map(|r| *r as i32).collect();
            let ext_comm = comm_new_subset(n, c_ranks.as_ptr());
            if ext_comm.is_null() {
                return Err("MPI failed to create subset communicator");
            }
            Ok(Communicator { handle: ext_comm })
        }
    }

    /// Terminates the MPI execution environment
    pub fn abort(&mut self, error_code: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_abort(self.handle, error_code);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to abort");
            }
        }
        Ok(())
    }

    /// Synchronizes the MPI processes
    pub fn barrier(&mut self) -> Result<(), StrError> {
        unsafe {
            let status = comm_barrier(self.handle);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to synchronize (barrier)");
            }
        }
        Ok(())
    }

    /// Determines the rank of the calling process in the communicator
    pub fn rank(&mut self) -> Result<usize, StrError> {
        let mut rank: i32 = 0;
        unsafe {
            let status = comm_rank(self.handle, &mut rank);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to get the rank");
            }
        }
        Ok(rank as usize)
    }

    /// Returns the size of the group associated with a communicator
    pub fn size(&mut self) -> Result<usize, StrError> {
        let mut size: i32 = 0;
        unsafe {
            let status = comm_size(self.handle, &mut size);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to get the size");
            }
        }
        Ok(size as usize)
    }

    //  broadcast --------------------------------------------------------------------------------------

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_i32(&mut self, sender: usize, x: &mut [i32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::I32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast i32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_i64(&mut self, sender: usize, x: &mut [i64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::I64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast i64 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_u32(&mut self, sender: usize, x: &mut [u32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_u64(&mut self, sender: usize, x: &mut [u64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u64 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    #[cfg(target_pointer_width = "32")]
    pub fn broadcast_usize(&mut self, sender: usize, x: &mut [usize]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.ext_comm, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast usize array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    #[cfg(target_pointer_width = "64")]
    pub fn broadcast_usize(&mut self, sender: usize, x: &mut [usize]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast usize array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_f32(&mut self, sender: usize, x: &mut [f32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::F32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast f32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_f64(&mut self, sender: usize, x: &mut [f64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::F64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast f64 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_c32(&mut self, sender: usize, x: &mut [Complex32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::C32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast Complex32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_c64(&mut self, sender: usize, x: &mut [Complex64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::C64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast Complex64 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_bytes(&mut self, sender: usize, x: &mut [u8]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::BYT.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast bytes array");
            }
        }
        Ok(())
    }

    // reduce -----------------------------------------------------------------------------------------

    /// Reduces values on all processes within a group
    pub fn reduce_i32(&mut self, root: usize, dest: &mut [i32], orig: &[i32], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce i32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_i64(&mut self, root: usize, dest: &mut [i64], orig: &[i64], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce i64 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_u32(&mut self, root: usize, dest: &mut [u32], orig: &[u32], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce u32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_u64(&mut self, root: usize, dest: &mut [u64], orig: &[u64], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce u64 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    #[cfg(target_pointer_width = "32")]
    pub fn reduce_usize(&mut self, root: usize, dest: &mut [usize], orig: &[usize], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce usize array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    #[cfg(target_pointer_width = "64")]
    pub fn reduce_usize(&mut self, root: usize, dest: &mut [usize], orig: &[usize], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce usize array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_f32(&mut self, root: usize, dest: &mut [f32], orig: &[f32], op: MpiOpReal) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce f32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_f64(&mut self, root: usize, dest: &mut [f64], orig: &[f64], op: MpiOpReal) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce f64 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_c32(&mut self, root: usize, dest: &mut [Complex32], orig: &[Complex32], op: MpiOpComplex) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce Complex32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_c64(&mut self, root: usize, dest: &mut [Complex64], orig: &[Complex64], op: MpiOpComplex) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce Complex64 array");
            }
        }
        Ok(())
    }

    // allreduce -----------------------------------------------------------------------------------------

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_i32(&mut self, dest: &mut [i32], orig: &[i32], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce i32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_i64(&mut self, dest: &mut [i64], orig: &[i64], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce i64 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_u32(&mut self, dest: &mut [u32], orig: &[u32], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce u32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_u64(&mut self, dest: &mut [u64], orig: &[u64], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce u64 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    #[cfg(target_pointer_width = "32")]
    pub fn allreduce_usize(&mut self, dest: &mut [usize], orig: &[usize], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce usize array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    #[cfg(target_pointer_width = "64")]
    pub fn allreduce_usize(&mut self, dest: &mut [usize], orig: &[usize], op: MpiOpInt) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce usize array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_f32(&mut self, dest: &mut [f32], orig: &[f32], op: MpiOpReal) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce f32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_f64(&mut self, dest: &mut [f64], orig: &[f64], op: MpiOpReal) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce f64 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_c32(&mut self, dest: &mut [Complex32], orig: &[Complex32], op: MpiOpComplex) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce Complex32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_c64(&mut self, dest: &mut [Complex64], orig: &[Complex64], op: MpiOpComplex) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce Complex64 array");
            }
        }
        Ok(())
    }

    // getters -------------------------------------------------------------------------------------------

    /// Returns the status of the last receive call
    pub fn get_receive_status(&mut self) -> (i32, i32, i32) {
        let mut source = 0;
        let mut tag = 0;
        let mut error = 0;
        unsafe {
            comm_get_receive_status(self.handle, &mut source, &mut tag, &mut error);
        }
        (source, tag, error)
    }

    // send ----------------------------------------------------------------------------------------------

    /// Performs a standard-mode blocking send
    pub fn send_i32(&mut self, data: &[i32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::I32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send i32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_i64(&mut self, data: &[i64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::I64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send i64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_u32(&mut self, data: &[u32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send u32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_u64(&mut self, data: &[u64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send u64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    #[cfg(target_pointer_width = "32")]
    pub fn send_usize(&mut self, data: &[usize], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    #[cfg(target_pointer_width = "64")]
    pub fn send_usize(&mut self, data: &[usize], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_f32(&mut self, data: &[f32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::F32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send f32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_f64(&mut self, data: &[f64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::F64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send f64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_c32(&mut self, data: &[Complex32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::C32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send Complex32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_c64(&mut self, data: &[Complex64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::C64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send Complex64 array");
            }
        }
        Ok(())
    }

    // receive -------------------------------------------------------------------------------------------

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_i32(&mut self, data: &mut [i32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::I32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive i32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_i64(&mut self, data: &mut [i64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::I64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive i64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_u32(&mut self, data: &mut [u32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive u32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_u64(&mut self, data: &mut [u64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive u64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    #[cfg(target_pointer_width = "32")]
    pub fn receive_usize(&mut self, data: &mut [usize], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    #[cfg(target_pointer_width = "64")]
    pub fn receive_usize(&mut self, data: &mut [usize], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_f32(&mut self, data: &mut [f32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::F32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive f32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_f64(&mut self, data: &mut [f64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::F64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive f64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_c32(&mut self, data: &mut [Complex32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::C32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive Complex32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_c64(&mut self, data: &mut [Complex64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::C64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive Complex64 array");
            }
        }
        Ok(())
    }

    // gather -------------------------------------------------------------------------------------------

    pub fn gather_i32(&mut self, root: usize, dest: Option<&mut [i32]>, orig: &[i32]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I32.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::I32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather i32 arrays");
            }
        }
        Ok(())
    }

    pub fn gather_i64(&mut self, root: usize, dest: Option<&mut [i64]>, orig: &[i64]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I64.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::I64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather i64 arrays");
            }
        }
        Ok(())
    }

    pub fn gather_u32(&mut self, root: usize, dest: Option<&mut [u32]>, orig: &[u32]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::U32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather u32 arrays");
            }
        }
        Ok(())
    }

    pub fn gather_u64(&mut self, root: usize, dest: Option<&mut [u64]>, orig: &[u64]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::U64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather u64 arrays");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "32")]
    pub fn gather_usize(&mut self, root: usize, dest: Option<&mut [usize]>, orig: &[usize]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::U32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather usize arrays");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    pub fn gather_usize(&mut self, root: usize, dest: Option<&mut [usize]>, orig: &[usize]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::U64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather usize arrays");
            }
        }
        Ok(())
    }

    pub fn gather_f32(&mut self, root: usize, dest: Option<&mut [f32]>, orig: &[f32]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F32.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::F32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather f32 arrays");
            }
        }
        Ok(())
    }

    pub fn gather_f64(&mut self, root: usize, dest: Option<&mut [f64]>, orig: &[f64]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F64.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::F64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather f64 arrays");
            }
        }
        Ok(())
    }

    pub fn gather_c32(&mut self, root: usize, dest: Option<&mut [Complex32]>, orig: &[Complex32]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C32.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::C32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather Complex32 arrays");
            }
        }
        Ok(())
    }

    pub fn gather_c64(&mut self, root: usize, dest: Option<&mut [Complex64]>, orig: &[Complex64]) -> Result<(), StrError> {
        unsafe {
            let status = match dest {
                Some(d) => {
                    let size = self.size()?;
                    if d.len() != size * orig.len() {
                        return Err("dest.len() must equal the number of processors times orig.len()");
                    }
                    comm_gather_im_root(self.handle, to_i32(root), to_i32(orig.len()), d.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C64.n())
                }
                None => comm_gather_im_not_root(self.handle, to_i32(root), to_i32(orig.len()), orig.as_ptr() as *const c_void, MpiType::C64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather Complex64 arrays");
            }
        }
        Ok(())
    }

    // allgather -------------------------------------------------------------------------------------------

    pub fn allgather_i32(&mut self, dest: &mut [i32], orig: &[i32]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather i32 arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_i64(&mut self, dest: &mut [i64], orig: &[i64]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather i64 arrays");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "32")]
    pub fn allgather_usize(&mut self, dest: &mut [usize], orig: &[usize]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather usize arrays");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    pub fn allgather_usize(&mut self, dest: &mut [usize], orig: &[usize]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather usize arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_u32(&mut self, dest: &mut [u32], orig: &[u32]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather u32 arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_u64(&mut self, dest: &mut [u64], orig: &[u64]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather u64 arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_f32(&mut self, dest: &mut [f32], orig: &[f32]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather f32 arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_f64(&mut self, dest: &mut [f64], orig: &[f64]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather f64 arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_c32(&mut self, dest: &mut [Complex32], orig: &[Complex32]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather Complex32 arrays");
            }
        }
        Ok(())
    }

    pub fn allgather_c64(&mut self, dest: &mut [Complex64], orig: &[Complex64]) -> Result<(), StrError> {
        let size = self.size()?;
        if dest.len() != size * orig.len() {
            return Err("dest.len() must equal the number of processors times orig.len()");
        }
        unsafe {
            let status = comm_allgather(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::C64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to gather Complex64 arrays");
            }
        }
        Ok(())
    }

    // scatter -------------------------------------------------------------------------------------------

    pub fn scatter_i32(&mut self, root: usize, dest: &mut [i32], orig: Option<&[i32]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::I32.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::I32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter i32 array");
            }
        }
        Ok(())
    }

    pub fn scatter_i64(&mut self, root: usize, dest: &mut [i64], orig: Option<&[i64]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::I64.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::I64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter i64 array");
            }
        }
        Ok(())
    }

    pub fn scatter_u32(&mut self, root: usize, dest: &mut [u32], orig: Option<&[u32]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::U32.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::U32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter u32 array");
            }
        }
        Ok(())
    }

    pub fn scatter_u64(&mut self, root: usize, dest: &mut [u64], orig: Option<&[u64]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::U64.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::U64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter u64 array");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "32")]
    pub fn scatter_usize(&mut self, root: usize, dest: &mut [usize], orig: Option<&[usize]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::U32.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::U32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter usize array");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    pub fn scatter_usize(&mut self, root: usize, dest: &mut [usize], orig: Option<&[usize]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::U64.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::U64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter usize array");
            }
        }
        Ok(())
    }

    pub fn scatter_f32(&mut self, root: usize, dest: &mut [f32], orig: Option<&[f32]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::F32.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::F32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter f32 array");
            }
        }
        Ok(())
    }

    pub fn scatter_f64(&mut self, root: usize, dest: &mut [f64], orig: Option<&[f64]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::F64.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::F64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter f64 array");
            }
        }
        Ok(())
    }

    pub fn scatter_c32(&mut self, root: usize, dest: &mut [Complex32], orig: Option<&[Complex32]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::C32.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::C32.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter Complex32 array");
            }
        }
        Ok(())
    }

    pub fn scatter_c64(&mut self, root: usize, dest: &mut [Complex64], orig: Option<&[Complex64]>) -> Result<(), StrError> {
        unsafe {
            let status = match orig {
                Some(o) => {
                    let size = self.size()?;
                    if o.len() != size * dest.len() {
                        return Err("orig.len() must equal the number of processors times dest.len()");
                    }
                    comm_scatter_im_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_mut_ptr() as *mut c_void, o.as_ptr() as *const c_void, MpiType::C64.n())
                }
                None => comm_scatter_im_not_root(self.handle, to_i32(root), to_i32(dest.len()), dest.as_ptr() as *mut c_void, MpiType::C64.n()),
            };
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to scatter Complex64 array");
            }
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mpi_finalize, mpi_init, Communicator};

    #[test]
    fn essential_features_work() {
        mpi_init().unwrap();
        let mut comm = Communicator::new().unwrap();
        assert_eq!(comm.rank().unwrap(), 0);
        assert_eq!(comm.size().unwrap(), 1);
        mpi_finalize().unwrap();
    }
}
